//===- NameMangling.cpp - Implementation of GPU symbols mangling ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the GPU dialect name mangling pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/Transforms/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "xblang/Dialect/Parallel/IR/Parallel.h"
#include "xblang/Dialect/Parallel/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::par;

namespace mlir {
namespace par {
#define GEN_PASS_DEF_GPUNAMEMANGLING
#include "xblang/Dialect/Parallel/Transforms/Passes.h.inc"
} // namespace par
} // namespace mlir

namespace {
// Mangle the names of all the top symbols inside a GPUModuleOp from symbol to
// "__G<module name>_S<symbol name>", for all GPUModuleOps in a module.
class GpuNameMangling
    : public mlir::par::impl::GpuNameManglingBase<GpuNameMangling> {
public:
  using Base::Base;

  // Get the mangled name for the symbol.
  StringAttr getMangledName(StringAttr moduleName, StringAttr symbol);

  // Mangle all the definitions inside a particular GPUModuleOp.
  LogicalResult mangleNamesInModule(gpu::GPUModuleOp module);

  // Update all the symbol uses of a particular symbol inside the top module.
  // `symbolUses` is the range of symbol uses of the gpu.module name in the top
  // module symbol table.
  void updateSymbolUses(SymbolTable::UseRange &&symbolUses);

  void runOnOperation() final;
};
} // namespace

StringAttr GpuNameMangling::getMangledName(StringAttr moduleName,
                                           StringAttr symbol) {
  std::string name = "__G" + moduleName.str() + "_S" + symbol.str();
  return StringAttr::get(&getContext(), name);
}

LogicalResult GpuNameMangling::mangleNamesInModule(gpu::GPUModuleOp gpuModule) {
  SymbolTable synbolTable(gpuModule);
  for (auto &op : gpuModule.getBody()->getOperations()) {
    // Ignore external functions.
    if (auto fn = dyn_cast<FunctionOpInterface>(op))
      if (fn.isExternal())
        continue;
    if (auto symbol = dyn_cast<SymbolOpInterface>(op)) {
      auto mangledName =
          getMangledName(gpuModule.getNameAttr(), symbol.getNameAttr());

      // Replace all the symbol uses of `symbol` to its mangled name.
      if (failed(synbolTable.replaceAllSymbolUses(
              symbol.getNameAttr(), mangledName, &gpuModule.getRegion()))) {
        emitError(op.getLoc(), "Failed to replace the symbol name.");
        return failure();
      }

      // On symbol replacement success rename the symbol.
      synbolTable.setSymbolName(symbol, mangledName);
    }
  }
  return success();
}

void GpuNameMangling::updateSymbolUses(SymbolTable::UseRange &&symbolUses) {
  // All symbolUses correspond to a particular gpu.module name.
  for (auto symbolUse : symbolUses) {
    Operation *operation = symbolUse.getUser();
    SmallVector<std::pair<StringAttr, SymbolRefAttr>> symbolReferences;

    // Collect all references to the `symbol` in the attributes of the
    // operation.
    for (auto opAttr : operation->getAttrs()) {
      if (auto symbol = dyn_cast<SymbolRefAttr>(opAttr.getValue()))
        if (symbol == symbolUse.getSymbolRef())
          symbolReferences.push_back({opAttr.getName(), symbol});
    }

    // Update the symbol references.
    for (auto &[attrName, symbol] : symbolReferences) {
      auto nestedReferences = symbol.getNestedReferences();
      if (nestedReferences.size()) {
        SmallVector<FlatSymbolRefAttr> updatedReferences(nestedReferences);
        // Only the first nested reference was updated by the previous step,
        // thus we just update that one.
        updatedReferences[0] = FlatSymbolRefAttr::get(getMangledName(
            symbol.getRootReference(), nestedReferences[0].getRootReference()));
        operation->setAttr(
            attrName,
            SymbolRefAttr::get(symbol.getRootReference(), updatedReferences));
      }
    }
  }
}

void GpuNameMangling::runOnOperation() {
  auto module = getOperation();
  SmallVector<gpu::GPUModuleOp> gpuModules;
  // Collect all gpu.modules.
  module.walk([&gpuModules](gpu::GPUModuleOp op) { gpuModules.push_back(op); });
  SymbolTable moduleTable(module);

  // Mangle the names.
  for (auto gpuModule : gpuModules) {
    if (failed(mangleNamesInModule(gpuModule)))
      return signalPassFailure();
    if (auto symbolUses = moduleTable.getSymbolUses(gpuModule.getNameAttr(),
                                                    &module.getRegion()))
      updateSymbolUses(std::move(*symbolUses));
  }
}
