#include "xblang/Dialect/XBLang/Concretization/Concretization.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xblang/Dialect/XBLang/Concretization/Common.h"
#include "xblang/Dialect/XBLang/Concretization/Decl.h"
#include "xblang/Dialect/XBLang/Concretization/Expr.h"
#include "xblang/Dialect/XBLang/Concretization/Stmt.h"
#include "xblang/Dialect/XBLang/Transforms/Passes.h"
#include "xblang/Dialect/XBLang/Transforms/Patterns.h"
#include "xblang/Sema/TypeSystem.h"

namespace xblang {
namespace xb {
#define GEN_PASS_DEF_XBLANGCONCRETIZER
#include "xblang/Dialect/XBLang/Transforms/Passes.h.inc"

namespace {
class XBLangConcretizer
    : public impl::XBLangConcretizerBase<XBLangConcretizer> {
public:
  using impl::XBLangConcretizerBase<XBLangConcretizer>::XBLangConcretizerBase;

  XBLangConcretizer(XBLangTypeSystem &typeSystem) : typeSystem(&typeSystem) {}

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    assert(typeSystem);
    populateConcretizationPatterns(*typeSystem, patterns);
    FrozenRewritePatternSet patternSet(std::move(patterns));
    mlir::GreedyRewriteConfig config;
    auto result =
        mlir::applyPatternsAndFoldGreedily(getOperation(), patternSet, config);
    if (result.failed())
      assert(false && "XBLangConcretizer failed.");
  }

  XBLangTypeSystem *typeSystem{};
};
} // namespace

void populateConcretizationPatterns(XBLangTypeSystem &typeSystem,
                                    mlir::RewritePatternSet &patterns) {
  patterns.add<RangeForOpConcretization>(typeSystem, patterns.getContext());
  patterns.add<RangeForCollapseOpConcretization>(typeSystem,
                                                 patterns.getContext(), 2);
  patterns.add<ArrayOpConcretization>(typeSystem, patterns.getContext());
  patterns.add<ArrayViewOpConcretization>(typeSystem, patterns.getContext());
  patterns.add<BinaryOpConcretization>(typeSystem, patterns.getContext());
  patterns.add<CastOpConcretization>(typeSystem, patterns.getContext());
  patterns.add<UnaryOponcretization>(typeSystem, patterns.getContext());
  patterns.add<CallOpConcretization>(typeSystem, patterns.getContext());
  patterns.add<SelectOpConcretization>(typeSystem, patterns.getContext());
  patterns.add<ImplicitCastRewriter>(patterns.getContext());
}

void populateConcretizationPasses(XBLangTypeSystem &typeSystem,
                                  mlir::PassManager &pm) {
  pm.addPass(std::unique_ptr<Pass>(new XBLangConcretizer(typeSystem)));
}
} // namespace xb
} // namespace xblang
