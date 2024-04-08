#include "xblang/Dialect/Parallel/IR/Parallel.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "xblang/Dialect/Parallel/IR/Dialect.h"
#include "xblang/Dialect/XBLang/IR/ASMUtils.h"
#include "xblang/Dialect/XBLang/IR/Enums.h"
#include "xblang/Dialect/XBLang/IR/Type.h"
#include "xblang/Dialect/XBLang/IR/XBLang.h"

using namespace mlir;
using namespace mlir::par;

void ParDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "xblang/Dialect/Parallel/IR/Parallel.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "xblang/Dialect/Parallel/IR/ParallelTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "xblang/Dialect/Parallel/IR/ParallelAttributes.cpp.inc"
      >();
}

namespace {
ParseResult parseQueue(OpAsmParser &parser,
                       OpAsmParser::UnresolvedOperand &operand, Type &type) {
  if (succeeded(parser.parseOptionalLSquare())) {
    if (parser.parseOperand(operand) || parser.parseColonType(type) ||
        parser.parseRSquare())
      return failure();
    return success();
  }
  return success();
}

ParseResult parseQueue(OpAsmParser &parser,
                       std::optional<OpAsmParser::UnresolvedOperand> &operand,
                       Type &type) {
  OpAsmParser::UnresolvedOperand queue;
  auto result = parseQueue(parser, queue, type);
  if (succeeded(result))
    operand = queue;
  return result;
}

void printQueue(OpAsmPrinter &p, Operation *op, Value operand, Type type) {
  if (operand)
    p << "[" << operand << " : " << type << "]";
}

ParseResult
parseMapList(OpAsmParser &parser,
             SmallVectorImpl<OpAsmParser::UnresolvedOperand> &vars,
             SmallVectorImpl<Type> &varTypes,
             SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
             SmallVectorImpl<Type> &types, ArrayAttr &mappings,
             SmallVectorImpl<OpAsmParser::UnresolvedOperand> &queues,
             SmallVectorImpl<Type> &queuesTypes) {
  SmallVector<MapKindAttr> mapKinds;
  auto parseAttr = [&](MapKindAttr &attr) -> mlir::ParseResult {
    StringRef enumStr;
    SMLoc loc = parser.getCurrentLocation();
    if (parser.parseKeyword(&enumStr))
      return failure();
    if (std::optional<MapKind> enumValue = symbolizeEnum<MapKind>(enumStr)) {
      attr = MapKindAttr::get(parser.getContext(), *enumValue);
      return success();
    }
    return parser.emitError(loc, "invalid clause value: '") << enumStr << "'";
  };
  if (failed(parser.parseCommaSeparatedList([&]() {
        if (parseAttr(mapKinds.emplace_back()) || parser.parseColon() ||
            parser.parseOperand(operands.emplace_back()) ||
            parser.parseColonType(types.emplace_back()) ||
            parser.parseArrow() || parser.parseOperand(vars.emplace_back()) ||
            parser.parseColonType(varTypes.emplace_back()) ||
            parseQueue(parser, queues.emplace_back(),
                       queuesTypes.emplace_back()))
          return failure();
        return success();
      })))
    return failure();
  SmallVector<Attribute> _maps(mapKinds.begin(), mapKinds.end());
  mappings = ArrayAttr::get(parser.getContext(), _maps);
  return success();
}

void printMapList(OpAsmPrinter &p, Operation *op, OperandRange varOperands,
                  TypeRange varTypes, OperandRange operands, TypeRange types,
                  std::optional<ArrayAttr> mapKinds, OperandRange queues,
                  TypeRange queuesTypes) {
  for (unsigned i = 0, e = mapKinds->size(); i < e; ++i) {
    if (i != 0)
      p << ", ";
    p << stringifyEnum(mlir::dyn_cast<MapKindAttr>((*mapKinds)[i]).getValue())
      << " : " << operands[i] << " : " << operands[i].getType() << " -> "
      << varOperands[i] << " : " << varOperands[i].getType();
    if (queues[i])
      p << " [" << queues[i] << ":" << queues[i].getType() << "]";
  }
}
} // namespace

//===----------------------------------------------------------------------===//
// Par loop Op
//===----------------------------------------------------------------------===//

LoopInfo::LoopInfo(StringAttr varName, LoopComparatorOp cmpOp,
                   LoopStepOp stepOp)
    : varName(varName), cmpOp(cmpOp), stepOp(stepOp) {}

bool LoopInfo::operator==(const LoopInfo &info) const {
  return varName == info.varName && cmpOp == info.cmpOp &&
         stepOp == info.stepOp;
}

llvm::hash_code mlir::par::hash_value(const mlir::par::LoopInfo &value) {
  return llvm::hash_combine(value.varName, value.cmpOp, value.stepOp);
}

/// Convert `value` to a DictAttr.
Attribute
mlir::par::convertToAttribute(MLIRContext *ctx,
                              const SmallVector<par::LoopInfo> &value) {
  SmallVector<Attribute> info;
  for (const auto &v : value)
    info.push_back(DictionaryAttr::get(
        ctx, {NamedAttribute(StringAttr::get(ctx, "name"), v.varName),
              NamedAttribute(StringAttr::get(ctx, "cmp"),
                             LoopComparatorOpAttr::get(ctx, v.cmpOp)),
              NamedAttribute(StringAttr::get(ctx, "step"),
                             LoopStepOpAttr::get(ctx, v.stepOp))}));
  return ArrayAttr::get(ctx, info);
}

/// Convert `attr` from a DictAttr.
LogicalResult
mlir::par::convertFromAttribute(SmallVector<par::LoopInfo> &value,
                                Attribute attr,
                                function_ref<InFlightDiagnostic()> diagnostic) {
  assert(attr && "Attribute cannot be null");
  auto array = dyn_cast<ArrayAttr>(attr);
  for (auto attr : array.getValue()) {
    LoopInfo info;
    assert(attr && "Attribute cannot be null");
    auto dict = dyn_cast<DictionaryAttr>(attr);
    if (!dict)
      return diagnostic() << "Member is not a dict array.";
    if (auto attrRaw = dict.get("name")) {
      if (auto name = dyn_cast<StringAttr>(attrRaw))
        info.varName = name;
      else
        return diagnostic() << "var `name` is not a StringAttr.";
    } else
      return diagnostic() << "var `name` is not present.";
    if (auto attrRaw = dict.get("cmp")) {
      if (auto cmpOp = dyn_cast<LoopComparatorOpAttr>(attrRaw))
        info.cmpOp = cmpOp.getValue();
      else
        return diagnostic() << "`cmp` is not a LoopComparatorOpAttr.";
    } else
      return diagnostic() << "`cmp` is not present.";
    if (auto attrRaw = dict.get("step")) {
      if (auto stepOp = dyn_cast<LoopStepOpAttr>(attrRaw))
        info.stepOp = stepOp.getValue();
      else
        return diagnostic() << "`step` is not a LoopStepOpAttr.";
    } else
      return diagnostic() << "`step` is not present.";
  }
  return success();
}

LoopBuilder::LoopBuilder(StringRef varName, Value begin, Value end,
                         std::optional<Location> varLoc, LoopComparatorOp cmpOp,
                         Value step, LoopStepOp stepOp)
    : varName(varName.str()), varLoc(varLoc), begin(begin), end(end),
      step(step), cmpOp(cmpOp), stepOp(stepOp) {}

void LoopOp::build(::mlir::OpBuilder &odsBuilder,
                   ::mlir::OperationState &odsState,
                   ArrayRef<LoopBuilder> loops,
                   ParallelHierarchy parallelization) {
  size_t numLoops = loops.size();
  SmallVector<Value> operands(numLoops * 3);
  SmallVector<LoopInfo> loopInfo(numLoops);
  SmallVector<Type> types(numLoops);
  SmallVector<Location> locs(numLoops, odsBuilder.getUnknownLoc());
  for (const auto &[i, loop] : llvm::enumerate(loops)) {
    assert(loop.begin && "`begin` cannot be null.");
    operands[i] = loop.begin;
    assert(loop.end && "`end` cannot be null.");
    operands[i + numLoops] = loop.end;
    operands[i + numLoops * 2] =
        loop.step ? loop.step
                  : odsBuilder.create<xblang::xb::ConstantOp>(
                        odsState.location,
                        odsBuilder.getIntegerAttr(loop.begin.getType(), 1));
    loopInfo[i] = LoopInfo(odsBuilder.getStringAttr(loop.varName), loop.cmpOp,
                           loop.stepOp);
    types[i] = loop.begin.getType();
    locs[i] = loop.varLoc ? *loop.varLoc : loop.begin.getLoc();
  }
  odsState.addOperands(operands);
  odsState.addAttribute(
      getParallelizationAttrName(odsState.name),
      odsBuilder.getAttr<ParallelHierarchyAttr>(parallelization));
  auto &props = odsState.getOrAddProperties<Properties>();
  props.loopInfo = std::move(loopInfo);
  auto region = odsState.addRegion();
  region->push_back(new mlir::Block());
  region->addArguments(TypeRange(types), locs);
}

LoopDescriptor LoopOp::getLoop(unsigned indx) {
  return LoopDescriptor({getRegion().getArgument(indx), getBegin()[indx],
                         getEnd()[indx], getStep()[indx],
                         getProperties().loopInfo[indx]});
}

SmallVector<LoopDescriptor> LoopOp::getLoops() {
  SmallVector<LoopDescriptor> loops;
  unsigned numLoops = getRegion().getNumArguments();
  loops.reserve(numLoops);
  for (unsigned i = 0; i < numLoops; ++i)
    loops.push_back(getLoop(i));
  return loops;
}

void LoopOp::getAsmBlockArgumentNames(mlir::Region &region,
                                      OpAsmSetValueNameFn setNameFn) {
  unsigned numVars = region.getNumArguments();
  auto &props = getProperties();
  for (unsigned i = 0; i < numVars; ++i)
    setNameFn(region.getArgument(i), props.loopInfo[i].varName.getValue());
}

LogicalResult LoopOp::verify() {
  for (auto [begin, end, step] : llvm::zip(getBegin(), getEnd(), getStep())) {
    if (begin.getType() != end.getType() || begin.getType() != step.getType())
      return failure();
  }
  return success();
}

SmallVector<mlir::Region *> LoopOp::getLoopRegions() { return {&getBody()}; }

void LoopOp::getSuccessorRegions(RegionBranchPoint point,
                                 SmallVectorImpl<RegionSuccessor> &regions) {
  regions.push_back(RegionSuccessor(&getBody()));
  regions.push_back(RegionSuccessor());
}

namespace {
ParseResult parseLoop(OpAsmParser &parser,
                      SmallVectorImpl<OpAsmParser::UnresolvedOperand> &begin,
                      SmallVectorImpl<Type> &beginTypes,
                      SmallVectorImpl<OpAsmParser::UnresolvedOperand> &end,
                      SmallVectorImpl<Type> &endTypes,
                      SmallVectorImpl<OpAsmParser::UnresolvedOperand> &step,
                      SmallVectorImpl<Type> &stepTypes,
                      LoopOp::Properties::loopInfoTy &props, Region &region) {
  SmallVector<OpAsmParser::Argument> vars;
  auto parseSingleLoop = [&]() -> LogicalResult {
    LoopInfo info;
    OpAsmParser::Argument var;
    if (parser.parseArgument(var, true) || parser.parseKeyword("in") ||
        parser.parseOperand(begin.emplace_back()) || parser.parseColon())
      return failure();
    if (succeeded(parser.parseOptionalLSquare())) {
      std::string stepOp;
      if (parser.parseString(&stepOp))
        return failure();
      if (auto op = stringToLoopStep(stepOp))
        info.stepOp = *op;
      else
        return parser.emitError(parser.getCurrentLocation(),
                                "Expected a Loop Step Operator.");
      if (parser.parseRSquare())
        return failure();
    }
    if (parser.parseOperand(step.emplace_back()) || parser.parseColon())
      return failure();
    if (succeeded(parser.parseOptionalLSquare())) {
      std::string cmpOp;
      if (parser.parseString(&cmpOp))
        return failure();
      if (auto op = stringToLoopComparator(cmpOp))
        info.cmpOp = *op;
      else
        return parser.emitError(parser.getCurrentLocation(),
                                "Expected a Loop Compare Operator.");
      if (parser.parseRSquare())
        return failure();
    }
    if (parser.parseOperand(end.emplace_back()))
      return failure();
    beginTypes.push_back(var.type);
    endTypes.push_back(var.type);
    stepTypes.push_back(var.type);
    info.varName = parser.getBuilder().getStringAttr(var.ssaName.name);
    props.push_back(info);
    vars.push_back(var);
    return success();
  };
  if (parser.parseLParen())
    return failure();
  if (failed(parseSingleLoop()))
    return failure();
  while (succeeded(parser.parseOptionalComma())) {
    if (failed(parseSingleLoop()))
      return failure();
  }
  if (parser.parseRParen())
    return failure();
  if (parser.parseRegion(region, vars))
    return failure();
  return xblang::xb::maybeEnsureFallthroughYield(parser, region);
}

void printLoop(OpAsmPrinter &p, Operation *op, OperandRange begin,
               TypeRange beginTypes, OperandRange end, TypeRange endTypes,
               OperandRange step, TypeRange stepTypes,
               const SmallVector<LoopInfo> &loops, Region &region) {
  p << "(";
  for (size_t i = 0; i < loops.size(); ++i) {
    p << region.getArgument(i) << " : " << region.getArgument(i).getType()
      << " in " << begin[i] << " : ";
    if (loops[i].stepOp != LoopStepOp::Add)
      p << "[\"" << stringifyEnum(loops[i].stepOp) << "\"] ";
    p << step[i] << " : ";
    if (loops[i].cmpOp != LoopComparatorOp::Less)
      p << "[\"" << stringifyEnum(loops[i].cmpOp) << "\"] ";
    p << end[i];
    if (i + 1 < loops.size())
      p << ", ";
  }
  p << ") ";
  xblang::xb::printRegionWithImplicitYield(p, op, region, false);
}
} // namespace

//===----------------------------------------------------------------------===//
// Par region Op
//===----------------------------------------------------------------------===//

void RegionOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                     ValueRange tensorDim, ValueRange matrixDim, Value queue,
                     ValueRange firstprivateVars, ValueRange privateVars,
                     ValueRange sharedVars, bool addBlock,
                     DataSharingKind defaultDataSharing,
                     std::pair<TypeRange, ArrayRef<Location>> matrixVars,
                     std::pair<TypeRange, ArrayRef<Location>> tensorVars,
                     ArrayRef<StringRef> varNames, ValueRange mappedVars,
                     ValueRange varMappings, ArrayAttr firstprivateVarsAttrs,
                     ArrayAttr privateVarsAttrs, ArrayAttr sharedVarsAttrs) {
  build(odsBuilder, odsState, tensorDim, matrixDim, defaultDataSharing,
        firstprivateVars, firstprivateVarsAttrs, privateVars, privateVarsAttrs,
        sharedVars, sharedVarsAttrs, mappedVars, varMappings, queue,
        odsBuilder.getStrArrayAttr(varNames));
  auto &props = odsState.getOrAddProperties<Properties>();
  props.numMatrixVars = matrixVars.first.size();
  if (addBlock) {
    Region &body = *odsState.regions.front();
    body.push_back(new Block);
    if (matrixVars.first.size() || tensorVars.first.size()) {
      body.addArguments(matrixVars.first, matrixVars.second);
      body.addArguments(tensorVars.first, tensorVars.second);
    }
  }
}

int64_t RegionOp::getNumMatrixVariables() {
  return getProperties().numMatrixVars;
}

int64_t RegionOp::getNumTensorVariables() {
  return getBody().getNumArguments() - getNumMatrixVariables();
}

ArrayRef<BlockArgument> RegionOp::getMatrixVariables() {
  return {getBody().args_begin(),
          std::next(getBody().args_begin(), getNumMatrixVariables())};
}

ArrayRef<BlockArgument> RegionOp::getTensorVariables() {
  return {std::next(getBody().args_begin(), getNumMatrixVariables()),
          getBody().args_end()};
}

BlockArgument RegionOp::addMatrixVariable(Type type, Location loc) {
  return getBody().insertArgument(
      std::next(getBody().args_begin(), getNumMatrixVariables()), type, loc);
}

BlockArgument RegionOp::addTensorVariable(Type type, Location loc) {
  return getBody().insertArgument(getBody().args_end(), type, loc);
}

void RegionOp::addFirstprivateVariable(ValueRange variables) {
  getFirstPrivateVarsMutable().append(variables);
}

void RegionOp::addPrivateVariable(ValueRange variables) {
  getPrivateVarsMutable().append(variables);
}

void RegionOp::addSharedVariable(ValueRange variables) {
  getSharedVarsMutable().append(variables);
}

void RegionOp::addMappings(ValueRange variables, ValueRange mappings) {
  getMappedVarsMutable().append(variables);
  getVarMappingsMutable().append(mappings);
}

void RegionOp::getAsmBlockArgumentNames(mlir::Region &region,
                                        OpAsmSetValueNameFn setNameFn) {
  unsigned numVars = region.getNumArguments();
  auto names = getProperties().attributionsNames.getValue();
  for (unsigned i = 0; i < numVars; ++i)
    setNameFn(region.getArgument(i), dyn_cast<StringAttr>(names[i]).getValue());
}

namespace {
ParseResult
parseParRegionArgs(OpAsmParser &parser,
                   SmallVectorImpl<OpAsmParser::UnresolvedOperand> &vars,
                   SmallVectorImpl<Type> &varsTy, ArrayAttr &argsAttrs) {
  SmallVector<OpAsmParser::Argument, 12> args;
  if (parser.parseArgumentList(args, AsmParser::Delimiter::Paren, true, true))
    return failure();
  SmallVector<Attribute, 12> attrs;
  for (auto &arg : args) {
    vars.emplace_back(arg.ssaName);
    varsTy.emplace_back(arg.type);
    attrs.emplace_back(arg.attrs);
  }
  argsAttrs = parser.getBuilder().getArrayAttr(attrs);
  return success();
}

void printParRegionArgs(OpAsmPrinter &p, Operation *op, ValueRange vars,
                        TypeRange varsTy, ArrayAttr varsAttrs) {
  p << "(";
  llvm::interleaveComma(
      llvm::iota_range<size_t>(0, vars.size(), false), p, [&](size_t i) {
        p << vars[i] << " : " << varsTy[i];
        if (varsAttrs && varsAttrs.size() > i)
          if (DictionaryAttr attrs =
                  dyn_cast_or_null<DictionaryAttr>(varsAttrs[i]))
            p.printOptionalAttrDict(attrs.getValue());
      });
  p << ")";
}

ParseResult parseParRegionBody(OpAsmParser &parser, Region &body,
                               int64_t &numMatrixVars, ArrayAttr &varIds) {
  SmallVector<OpAsmParser::Argument, 12> vars;
  if (succeeded(parser.parseOptionalKeyword("matrix_vars"))) {
    if (parser.parseArgumentList(vars, AsmParser::Delimiter::Paren, true))
      return failure();
  }
  numMatrixVars = vars.size();
  if (succeeded(parser.parseOptionalKeyword("tensor_vars"))) {
    if (parser.parseArgumentList(vars, AsmParser::Delimiter::Paren, true))
      return failure();
  }
  SmallVector<Attribute, 12> varNames;
  for (auto &var : vars)
    varNames.emplace_back(parser.getBuilder().getStringAttr(var.ssaName.name));
  if (parser.parseRegion(body, vars))
    return failure();
  varIds = parser.getBuilder().getArrayAttr(varNames);
  return xblang::xb::maybeEnsureFallthroughYield(parser, body);
}

void printParRegionBody(OpAsmPrinter &p, RegionOp op, Region &body,
                        int64_t numMatrixVars, ArrayAttr varIds) {
  if (numMatrixVars) {
    p << "matrix_vars(";
    for (auto var : op.getMatrixVariables())
      p.printRegionArgument(var);
    p << ") ";
  }
  if (op.getNumTensorVariables()) {
    p << "tensor_vars(";
    for (auto var : op.getTensorVariables())
      p.printRegionArgument(var);
    p << ") ";
  }
  xblang::xb::printRegionWithImplicitYield(p, op, body, false);
}

ParseResult
parseRegionMappings(OpAsmParser &parser,
                    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &vars,
                    SmallVectorImpl<Type> &varsTy,
                    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &mappings,
                    SmallVectorImpl<Type> &mappingsTy) {
  auto parseMapping = [&]() -> ParseResult {
    if (parser.parseOperand(vars.emplace_back()) ||
        parser.parseColonType(varsTy.emplace_back()) || parser.parseEqual() ||
        parser.parseOperand(mappings.emplace_back()) ||
        parser.parseColonType(mappingsTy.emplace_back()))
      return failure();
    return success();
  };
  if (parser.parseLParen())
    return failure();
  do {
    if (parseMapping())
      return failure();
  } while (succeeded(parser.parseOptionalComma()));
  if (parser.parseRParen())
    return failure();
  return success();
}

void printRegionMappings(OpAsmPrinter &p, Operation *op, ValueRange vars,
                         TypeRange varsTy, ValueRange mappings,
                         TypeRange mappingsTy) {
  if (mappings.empty())
    return;
  p << "(";
  llvm::interleaveComma(llvm::iota_range<size_t>(0, mappings.size(), false), p,
                        [&](size_t i) {
                          p << vars[i] << " : " << varsTy[i] << " = "
                            << mappings[i] << " : " << mappingsTy[i];
                        });
  p << ")";
}
} // namespace

//===----------------------------------------------------------------------===//
// Par func Op
//===----------------------------------------------------------------------===//

LogicalResult FuncOp::verifyType() {
  auto type = getFunctionType();
  if (!type.isa<FunctionType>())
    return emitOpError("requires '" + getFunctionTypeAttrName().str() +
                       "' attribute of function type");
  if (getFunctionType().getNumResults() > 1)
    return emitOpError("cannot have more than one result");
  return success();
}

LogicalResult FuncOp::verify() { return verifyType(); }

void FuncOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   llvm::StringRef name, mlir::FunctionType type,
                   llvm::ArrayRef<mlir::NamedAttribute> attrs,
                   llvm::ArrayRef<DictionaryAttr> argAttrs) {
  buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());

  if (argAttrs.empty())
    return;
  assert(type.getNumInputs() == argAttrs.size());
  function_interface_impl::addArgAndResultAttrs(
      builder, state, argAttrs, /*resultAttrs=*/std::nullopt,
      getArgAttrsAttrName(state.name), getResAttrsAttrName(state.name));
}

mlir::ParseResult FuncOp::parse(mlir::OpAsmParser &parser,
                                mlir::OperationState &result) {
  auto fnBuilder =
      [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
         llvm::ArrayRef<mlir::Type> results,
         mlir::function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };
  return mlir::function_interface_impl::parseFunctionOp(
      parser, result, false, getFunctionTypeAttrName(result.name), fnBuilder,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FuncOp::print(mlir::OpAsmPrinter &p) {
  mlir::function_interface_impl::printFunctionOp(
      p, *this, false, getFunctionTypeAttrName(), getArgAttrsAttrName(),
      getResAttrsAttrName());
}

OpFoldResult DefaultQueueOp::fold(FoldAdaptor adaptor) {
  return adaptor.getAttributes();
}

#include "xblang/Dialect/Parallel/IR/ParallelDialect.cpp.inc"

#include "xblang/Dialect/Parallel/IR/ParallelEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "xblang/Dialect/Parallel/IR/ParallelTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "xblang/Dialect/Parallel/IR/ParallelAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "xblang/Dialect/Parallel/IR/Parallel.cpp.inc"
