#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xblang/Dialect/Parallel/IR/Parallel.h"
#include "xblang/Dialect/Parallel/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::par;

namespace mlir {
namespace par {
#define GEN_PASS_DEF_OFFLOADMERGEBINARIES
#include "xblang/Dialect/Parallel/Transforms/Passes.h.inc"
} // namespace par
} // namespace mlir

namespace {
class OffloadMergeBinaries
    : public mlir::par::impl::OffloadMergeBinariesBase<OffloadMergeBinaries> {
public:
  using Base::Base;
  void runOnOperation() final;
  void updateSymbolUses(StringAttr binName, SymbolTable::UseRange &&symbolUses);
};
} // namespace

void OffloadMergeBinaries::updateSymbolUses(
    StringAttr binName, SymbolTable::UseRange &&symbolUses) {
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
    for (auto &[attrName, symbol] : symbolReferences)
      operation->setAttr(
          attrName, SymbolRefAttr::get(binName, symbol.getNestedReferences()));
  }
}

void OffloadMergeBinaries::runOnOperation() {
  OpBuilder builder(&getContext());
  builder.setInsertionPointToStart(getOperation().getBody());
  SmallVector<gpu::BinaryOp> binaries;
  for (Region &region : getOperation()->getRegions())
    for (Block &block : region.getBlocks())
      for (auto binary : block.getOps<gpu::BinaryOp>())
        binaries.push_back(binary);
  SmallVector<Attribute> objects;
  Attribute manager{};
  for (gpu::BinaryOp bin : binaries) {
    auto objs = bin.getObjects().getValue();
    objects.append(objs.begin(), objs.end());
    if (!manager)
      manager = bin.getOffloadingHandler().value();
    if (manager != bin.getOffloadingHandler()) {
      getOperation().emitError() << "All offloading handler must be the same!";
      return signalPassFailure();
    }
  }
  SymbolTable table(getOperation());
  auto binName = builder.getStringAttr("offloading_kernels");
  builder.create<gpu::BinaryOp>(getOperation().getLoc(), binName.getValue(),
                                manager, objects);
  for (gpu::BinaryOp bin : binaries) {
    if (auto symbolUses =
            table.getSymbolUses(bin.getNameAttr(), &getOperation().getRegion()))
      updateSymbolUses(binName, std::move(*symbolUses));
    bin.erase();
  }
}
