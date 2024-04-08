#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xblang/Dialect/XBLang/IR/XBLang.h"
#include "xblang/Dialect/XBLang/Transforms/Patterns.h"
#include <memory>

namespace xblang {
namespace xb {
#define GEN_PASS_DEF_IMPLICITCASTCONCRETIZER
#include "xblang/Dialect/XBLang/Transforms/Passes.h.inc"

LogicalResult
ImplicitCastRewriter::matchAndRewrite(Interface interface,
                                      PatternRewriter &rewriter) const {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(interface);
  SmallVector<std::pair<size_t, mlir::Value>> args;
  for (auto arg : llvm::enumerate(interface->getOperands()))
    if (auto type = interface.getImplicitCast(arg.index())) {
      mlir::Value value = arg.value();
      if (type != value.getType()) {
        value = rewriter.create<CastOp>(value.getLoc(), type, value);
        args.push_back({arg.index(), value});
      }
    }
  if (args.size()) {
    rewriter.modifyOpInPlace(interface, [&args, interface]() {
      for (auto arg : args)
        interface->setOperand(arg.first, arg.second);
    });
    return success();
  }
  return failure();
}

class ImplicitCastConcretizer
    : public impl::ImplicitCastConcretizerBase<ImplicitCastConcretizer> {
public:
  using impl::ImplicitCastConcretizerBase<
      ImplicitCastConcretizer>::ImplicitCastConcretizerBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<ImplicitCastRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    auto result =
        mlir::applyPatternsAndFoldGreedily(getOperation(), patternSet);
    if (result.failed())
      assert(false && "ImplicitCastConcretizer failed.");
  }
};
} // namespace xb
} // namespace xblang
