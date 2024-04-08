#include "xblang/Dialect/Parallel/IR/Parallel.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "xblang/Lang/Parallel/Frontend/Options.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Object/OffloadBinary.h"

using namespace mlir;
using namespace mlir::par;

namespace llvm {
namespace {
struct OffloadManager {
  static LogicalResult
  embedBinary(gpu::BinaryOp op, llvm::IRBuilderBase &hostBuilder,
              LLVM::ModuleTranslation &hostModuleTranslation);

  static LogicalResult
  launchKernel(gpu::LaunchFuncOp launchFuncOp, llvm::IRBuilderBase &hostBuilder,
               LLVM::ModuleTranslation &hostModuleTranslation);

  static void
  createKernelCall(mlir::gpu::LaunchFuncOp op, IRBuilderBase &builder,
                   mlir::LLVM::ModuleTranslation &hostModuleTranslation);
  // Get intptr_t;
  static IntegerType *getSizeTTy(Module &M);

  // Get the type of the offloading entry.
  static StructType *getEntryTy(Module &M);

  // Merge all binaries in a binary op.
  static std::optional<SmallVector<char, 1024>>
  mergeBinaries(mlir::gpu::BinaryOp op, llvm::IRBuilderBase &hostBuilder);
};
} // namespace
} // namespace llvm

llvm::IntegerType *llvm::OffloadManager::getSizeTTy(Module &M) {
  LLVMContext &C = M.getContext();
  return M.getDataLayout().getIntPtrType(C);
}

llvm::StructType *llvm::OffloadManager::getEntryTy(Module &M) {
  LLVMContext &C = M.getContext();
  StructType *EntryTy = StructType::getTypeByName(C, "__tgt_offload_entry");
  if (!EntryTy)
    EntryTy =
        StructType::create("__tgt_offload_entry", PointerType::getUnqual(C),
                           PointerType::getUnqual(C), getSizeTTy(M),
                           Type::getInt32Ty(C), Type::getInt32Ty(C));
  return EntryTy;
}

void llvm::OffloadManager::createKernelCall(
    mlir::gpu::LaunchFuncOp op, IRBuilderBase &builder,
    mlir::LLVM::ModuleTranslation &hostModuleTranslation) {
  auto args = op.getKernelOperands();
  auto grid = op.getGridSizeOperandValues();
  auto block = op.getBlockSizeOperandValues();
  auto dynamicMemorySz = op.getDynamicSharedMemorySize();
  auto asyncObject = op.getAsyncObject();
  auto llvmArgs = hostModuleTranslation.lookupValues(args);
  llvm::Module &hostModule = *hostModuleTranslation.getLLVMModule();

  SmallVector<Type *> structTypes;
  for (auto arg : llvmArgs) {
    assert(arg);
    structTypes.push_back(arg->getType());
  }

  auto structTy = StructType::create(structTypes);
  auto argStruct = builder.CreateAlloca(StructType::create(structTy), 0u);
  auto argArray = builder.CreateAlloca(
      builder.getPtrTy(0),
      ConstantInt::get(getSizeTTy(hostModule), structTypes.size()));

  for (auto [i, arg] : enumerate(llvmArgs)) {
    auto structMember = builder.CreateStructGEP(structTy, argStruct, i);
    builder.CreateStore(arg, structMember);
    auto arrayMember =
        builder.CreateConstGEP1_32(builder.getPtrTy(0), argArray, i);
    builder.CreateStore(structMember, arrayMember);
  }

  auto llVal = [&](mlir::Value value) -> Value * {
    auto val = hostModuleTranslation.lookupValue(value);
    assert(val);
    return val;
  };

  auto ptrTy = builder.getPtrTy(0);
  auto intPtrTy = getSizeTTy(hostModule);
  auto i32Ty = builder.getInt32Ty();

  Value *smem =
      dynamicMemorySz ? llVal(dynamicMemorySz) : ConstantInt::get(i32Ty, 0);
  Value *gx = llVal(grid.x), *gy = llVal(grid.y), *gz = llVal(grid.z);
  Value *bx = llVal(block.x), *by = llVal(block.y), *bz = llVal(block.z);
  Value *nullPtr = ConstantPointerNull::get(ptrTy);
  Value *stream = nullPtr;
  if (asyncObject)
    stream = llVal(asyncObject);

  auto kernelName = op.getKernelName().getValue();
  std::string stubName = kernelName.str() + "_stub";
  auto fnCallee = hostModule.getOrInsertFunction(
      stubName, llvm::FunctionType::get(builder.getVoidTy(), false));

  auto launchKernel = hostModule.getOrInsertFunction(
      "__xblangLaunch",
      llvm::FunctionType::get(
          builder.getVoidTy(),
          ArrayRef<Type *>({ptrTy, intPtrTy, intPtrTy, intPtrTy, intPtrTy,
                            intPtrTy, intPtrTy, i32Ty, ptrTy, ptrTy, ptrTy}),
          false));
  builder.CreateCall(
      launchKernel, ArrayRef<Value *>({fnCallee.getCallee(), gx, gy, gz, bx, by,
                                       bz, smem, stream, argArray, nullPtr}));
}

std::optional<SmallVector<char, 1024>>
llvm::OffloadManager::mergeBinaries(mlir::gpu::BinaryOp op,
                                    llvm::IRBuilderBase &hostBuilder) {
  using namespace object;
  OffloadAttr offAttr = dyn_cast<OffloadAttr>(op.getOffloadingHandlerAttr());
  SmallVector<char, 1024> offloadData;
  raw_svector_ostream outputStream(offloadData);
  ImageKind imageKind = IMG_Object;
  auto appendObject = [&](StringRef binary, ImageKind imageKind,
                          OffloadKind offKind, StringRef triple,
                          StringRef chip) -> LogicalResult {
    // Create the offload object, for more information check:
    // https://clang.llvm.org/docs/ClangOffloadPackager.html
    SmallString<0> buffer;
    {
      OffloadBinary::OffloadingImage imageBinary{};
      imageBinary.TheImageKind = imageKind;
      imageBinary.TheOffloadKind = offKind;
      imageBinary.StringData["triple"] = triple;
      // Avoid setting the arch if no arch was given in the cmd, as clang will
      // compile code only for this arch if set, so running the code on an
      // incompatible arch will result in error.
      if (chip.size())
        imageBinary.StringData["arch"] = chip;
      imageBinary.Image = MemoryBuffer::getMemBuffer(binary, "", false);
      buffer = OffloadBinary::write(imageBinary);
    }
    // Check that the image was properly created. This step was taken from:
    // https://github.com/llvm/llvm-project/blob/main/clang/tools/clang-offload-packager/ClangOffloadPackager.cpp
    if (buffer.size() % OffloadBinary::getAlignment() != 0) {
      op.emitError("Offload binary has an invalid size alignment");
      return failure();
    }
    // Write the buffer.
    outputStream << buffer;
    return success();
  };
  if (offAttr.getKind() == "llvm")
    imageKind = IMG_Bitcode;
  for (auto objectAttr : op.getObjectsAttr().getValue()) {
    auto object = dyn_cast<gpu::ObjectAttr>(objectAttr);
    ImageKind imgKind = imageKind;
    OffloadKind offKind = OFK_None;
    StringRef triple, chip;
    if (auto nvptx = mlir::dyn_cast<NVVM::NVVMTargetAttr>(object.getTarget())) {
      triple = nvptx.getTriple();
      chip = nvptx.getChip();
      if (imageKind == IMG_Object)
        imgKind = IMG_Cubin;
      offKind = OFK_Cuda;
    } else if (auto amdgpu =
                   mlir::dyn_cast<ROCDL::ROCDLTargetAttr>(object.getTarget())) {
      triple = amdgpu.getTriple();
      chip = amdgpu.getChip();
      offKind = OFK_HIP;
    }
    if (failed(appendObject(object.getObject().getValue(), imgKind, offKind,
                            triple, chip)))
      return std::nullopt;
  }
  return offloadData;
}

LogicalResult llvm::OffloadManager::embedBinary(
    gpu::BinaryOp op, llvm::IRBuilderBase &hostBuilder,
    LLVM::ModuleTranslation &hostModuleTranslation) {
  llvm::Module *hostModule = hostModuleTranslation.getLLVMModule();

  if (llvm::GlobalVariable *gv =
          hostModule->getGlobalVariable("llvm.embedded.object", true)) {
    op.emitError("There should be only one binary.");
    return failure();
  } else {
    auto bin = mergeBinaries(op, hostBuilder);
    if (!bin)
      return failure();
    auto binConst = llvm::ConstantDataArray::getString(
        hostBuilder.getContext(), StringRef(bin->data(), bin->size()), false);
    llvm::GlobalVariable *serializedObj = new llvm::GlobalVariable(
        *hostModule, binConst->getType(), true,
        llvm::GlobalValue::LinkageTypes::PrivateLinkage, binConst,
        "llvm.embedded.object");

    serializedObj->setLinkage(llvm::GlobalValue::LinkageTypes::PrivateLinkage);
    serializedObj->setAlignment(llvm::MaybeAlign(8));
    serializedObj->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::None);
    serializedObj->setSection(".llvm.offloading");

    SmallVector<llvm::Constant *, 4u> ImagesInits({serializedObj});
    auto *ImagesData = llvm::ConstantArray::get(
        llvm::ArrayType::get(hostBuilder.getPtrTy(), 1), ImagesInits);
    llvm::GlobalVariable *usedVar = new llvm::GlobalVariable(
        *hostModule, ImagesData->getType(), true,
        llvm::GlobalValue::LinkageTypes::AppendingLinkage, ImagesData,
        "llvm.compiler.used");
    usedVar->setSection("llvm.metadata");
  }
  return success();
}

LogicalResult llvm::OffloadManager::launchKernel(
    gpu::LaunchFuncOp launchFuncOp, llvm::IRBuilderBase &hostBuilder,
    LLVM::ModuleTranslation &hostModuleTranslation) {
  llvm::Module *hostModule = hostModuleTranslation.getLLVMModule();
  auto kernelName = launchFuncOp.getKernelName().getValue();
  std::string stubName = kernelName.str() + "_stub";
  if (!hostModule->getFunction(stubName)) {

    auto fnCallee = hostModule->getOrInsertFunction(
        stubName, llvm::FunctionType::get(hostBuilder.getVoidTy(), false));

    auto fn = mlir::dyn_cast<llvm::Function>(fnCallee.getCallee());
    {
      llvm::IRBuilder<> builder(
          llvm::BasicBlock::Create(hostBuilder.getContext(), "entry", fn));
      builder.CreateRetVoid();
    }
    fn->setDSOLocal(true);

    llvm::GlobalVariable *entryName = hostBuilder.CreateGlobalString(
        kernelName, ".omp_offloading.entry_name." + kernelName, 0, hostModule);
    entryName->setLinkage(llvm::GlobalValue::LinkageTypes::InternalLinkage);

    auto entryType = getEntryTy(*hostModule);
    auto z32 = llvm::ConstantInt::get(hostBuilder.getInt32Ty(), 0);
    auto z64 = llvm::ConstantInt::get(getSizeTTy(*hostModule), 0);

    SmallVector<llvm::Constant *, 5> structElements;
    structElements.push_back(fn);
    structElements.push_back(entryName);
    structElements.push_back(z64);
    structElements.push_back(z32);
    structElements.push_back(z32);

    llvm::GlobalVariable *entry = new llvm::GlobalVariable(
        *hostModule, entryType, true,
        llvm::GlobalValue::LinkageTypes::WeakAnyLinkage,
        llvm::ConstantStruct::get(entryType, fn, entryName, z64, z32, z32),
        "omp_offloading.entry." + kernelName);
    ::xblang::par::ParOptions opts;
    if (opts.isNVPTX())
      entry->setSection("cuda_offloading_entries");
    else if (opts.isAMDGPU())
      entry->setSection("hip_offloading_entries");

    entry->setAlignment(llvm::MaybeAlign(1));
    entry->setLinkage(llvm::GlobalValue::LinkageTypes::WeakAnyLinkage);
    entry->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::None);
  }
  createKernelCall(launchFuncOp, hostBuilder, hostModuleTranslation);
  return success();
}

LogicalResult
OffloadAttr::embedBinary(Operation *op, llvm::IRBuilderBase &hostBuilder,
                         LLVM::ModuleTranslation &hostModuleTranslation) const {
  auto binaryOp = mlir::dyn_cast<gpu::BinaryOp>(op);
  assert(binaryOp && "Op is not a BinaryOp.");
  return llvm::OffloadManager::embedBinary(binaryOp, hostBuilder,
                                           hostModuleTranslation);
}

LogicalResult OffloadAttr::launchKernel(
    Operation *launchFuncOperation, Operation *binaryOperation,
    llvm::IRBuilderBase &hostBuilder,
    LLVM::ModuleTranslation &hostModuleTranslation) const {
  auto launchFuncOp = mlir::dyn_cast<gpu::LaunchFuncOp>(launchFuncOperation);
  assert(launchFuncOp && "Op is not a LaunchFuncOp.");
  return llvm::OffloadManager::launchKernel(launchFuncOp, hostBuilder,
                                            hostModuleTranslation);
}
