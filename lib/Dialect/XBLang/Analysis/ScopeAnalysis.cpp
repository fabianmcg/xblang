#include "xblang/Dialect/XBLang/Analysis/ScopeAnalysis.h"

#include "mlir/IR/OpDefinition.h"
#include <stack>

using namespace mlir;
using namespace xblang::xb;

namespace {
struct ScopedValueUseAnalysisImpl
    : public OpOperandAnalysisWalk<ScopedValueUseAnalysisImpl> {
  using ScopeInfo = ScopedValueUseAnalysis::ScopeInfo;

  ScopedValueUseAnalysisImpl(DenseMap<Operation *, ScopeInfo> &opUses)
      : opUses(opUses) {}

  inline void anaylizeOperand(Stack &analysisStack, OpOperand &operand,
                              Operation *definigOp) {
    using ItTy = Value::use_iterator;
    if (auto parentOp = definigOp->getParentOp())
      for (Operation *limitOp : llvm::reverse(analysisStack)) {
        if (parentOp->isProperAncestor(limitOp)) {
          auto &scopeInfo = opUses[limitOp];
          auto &scopeUses = scopeInfo[definigOp];
          if (!scopeUses)
            scopeUses = ScopeOpUses(ItTy(&operand), ItTy(&operand));
          scopeUses = ScopeOpUses(scopeUses.begin(), ++ItTy(&operand));
        } else
          break;
      }
  }

  DenseMap<Operation *, ScopeInfo> &opUses;
};

struct SimpleScopedValueUseAnalysisImpl
    : public OpOperandAnalysisWalk<SimpleScopedValueUseAnalysisImpl> {
  using UsesMap = SimpleScopedValueUseAnalysis::UsesMap;
  using UsersMap = SimpleScopedValueUseAnalysis::UsersMap;

  SimpleScopedValueUseAnalysisImpl(UsesMap &uses, UsersMap &users)
      : uses(uses), users(users) {}

  inline void anaylizeOperand(Stack &analysisStack, OpOperand &operand,
                              Operation *definigOp) {
    if (auto parentOp = definigOp->getParentOp())
      for (Operation *limitOp : llvm::reverse(analysisStack)) {
        if (parentOp->isProperAncestor(limitOp)) {
          uses[limitOp].insert(operand.get());
          users[operand.get()].insert(limitOp);
        } else
          break;
      }
  }

  UsesMap &uses;
  UsersMap &users;
};
} // namespace

ScopedValueUseAnalysis::ScopedValueUseAnalysis(Operation *topOp,
                                               OpFilter opFilter,
                                               OperandFilter operandFilter) {
  ScopedValueUseAnalysisImpl(opUses).run(topOp, opFilter, operandFilter);
}

SimpleScopedValueUseAnalysis::SimpleScopedValueUseAnalysis(
    Operation *topOp, OpFilter opFilter, OperandFilter operandFilter) {
  SimpleScopedValueUseAnalysisImpl(valueUses, valueUsers)
      .run(topOp, opFilter, operandFilter);
}
