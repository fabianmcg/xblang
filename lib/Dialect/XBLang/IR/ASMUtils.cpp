#include "xblang/Dialect/XBLang/IR/ASMUtils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "xblang/Dialect/XBLang/IR/XBLang.h"

using namespace mlir;
using namespace xblang::xb;

void xblang::xb::ensureTerminator(Region &region,
                                  TerminatorBuilder terminatorBuilder) {
  if (region.hasOneBlock() && !region.front().empty())
    if (!region.front().back().hasTrait<OpTrait::IsTerminator>()) {
      OpBuilder builder(&region.front(), region.front().end());
      terminatorBuilder(builder);
    }
}

void xblang::xb::maybeEnsureFallthroughYield(Region &region) {
  ensureTerminator(region, [&](OpBuilder &builder) {
    builder.create<YieldOp>(region.getLoc(), YieldKind::Fallthrough,
                            ValueRange());
  });
}

ParseResult xblang::xb::maybeEnsureFallthroughYield(OpAsmParser &parser,
                                                    Region &region) {
  if (region.hasOneBlock() && !region.front().empty())
    if (!region.front().back().hasTrait<OpTrait::IsTerminator>()) {
      OpBuilder builder(parser.getContext());
      builder.setInsertionPoint(&region.front(), region.front().end());
      builder.create<YieldOp>(
          parser.getEncodedSourceLoc(parser.getCurrentLocation()),
          YieldKind::Fallthrough, ValueRange());
    }
  return success();
}

ParseResult xblang::xb::parseRegionWithImplicitYield(OpAsmParser &parser,
                                                     Region &region) {
  if (parser.parseRegion(region))
    return failure();
  return maybeEnsureFallthroughYield(parser, region);
}

void xblang::xb::printRegionWithImplicitYield(OpAsmPrinter &printer,
                                              Operation *op, Region &region,
                                              bool printEntryBlockArgs,
                                              bool printEmptyBlock) {
  if (region.hasOneBlock() && !region.front().empty())
    if (auto yieldOp = dyn_cast<YieldOp>(region.front().getTerminator()))
      if (yieldOp.getKind() == YieldKind::Fallthrough &&
          yieldOp->getNumOperands() == 0) {
        printer.printRegion(region, printEntryBlockArgs, false,
                            printEmptyBlock);
        return;
      }
  printer.printRegion(region, printEntryBlockArgs, true, printEmptyBlock);
}
