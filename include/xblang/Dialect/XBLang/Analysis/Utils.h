#ifndef XBLANG_DIALECT_XBLANG_ANALYSIS_UTILS_H
#define XBLANG_DIALECT_XBLANG_ANALYSIS_UTILS_H

#include "mlir/IR/OpDefinition.h"
#include "xblang/Support/LLVM.h"

namespace xblang {
namespace xb {
class AnalysisWalkBase {
public:
  enum class Walk : uint8_t {
    analyze,
    skip,
    terminate,
    visit,
  };
  using OpFilter = llvm::function_ref<Walk(Operation *, bool)>;
  using OperandFilter = llvm::function_ref<bool(Operation *, OpOperand *)>;
};

template <typename Derived>
class OpOperandAnalysisWalk : public AnalysisWalkBase {
public:
  using Base = OpOperandAnalysisWalk;

  void run(Operation *rootOp, OpFilter opFilter, OperandFilter operandFilter) {
    SmallVector<Operation *, 16> analysisStack;
    SmallVector<Operation *, 32> walkStack;
    walkStack.push_back(rootOp);
    this->opFilter = opFilter;
    this->operandFilter = operandFilter;
    assert(opFilter && operandFilter && "Both filters must be valid.");
    (void)analyze(analysisStack, walkStack);
    this->opFilter = nullptr;
    this->operandFilter = nullptr;
  }

protected:
  using Stack = SmallVectorImpl<Operation *>;

  OpOperandAnalysisWalk() = default;

  /// Analyze the operations in analysisStack.
  Walk analyze(Stack &analysisStack, Stack &walkStack) {
    while (walkStack.size()) {
      auto topOp = walkStack.pop_back_val();
      if (!topOp)
        return Walk::skip;
      for (Region &region : topOp->getRegions())
        for (Block &block : region.getBlocks())
          for (Operation &op : block.getOperations()) {
            Walk walkStatus = opFilter(&op, analysisStack.size());
            if (walkStatus == Walk::skip)
              continue;
            else if (walkStatus == Walk::terminate)
              return walkStatus;
            if (analysisStack.size())
              getDerived().analyzeOperands(analysisStack, op);
            if (walkStatus == Walk::visit) {
              walkStack.push_back(&op);
              continue;
            }
            walkStack.push_back(nullptr);
            walkStack.push_back(&op);
            analysisStack.push_back(&op);
            walkStatus = analyze(analysisStack, walkStack);
            analysisStack.pop_back();
            if (walkStatus == Walk::terminate)
              return walkStatus;
          }
    }
    return Walk::skip;
  }

  /// Analyze the operands of op.
  inline void analyzeOperands(Stack &analysisStack, Operation &op) {
    for (OpOperand &operand : op.getOpOperands()) {
      auto definigOp = operand.get().getDefiningOp();
      if (!operandFilter(definigOp, &operand))
        continue;
      getDerived().anaylizeOperand(analysisStack, operand, definigOp);
    }
  }

  /// Analyze an operand.
  inline void anaylizeOperand(Stack &analysisStack, OpOperand &operand,
                              Operation *definigOp) {}

private:
  Derived &getDerived() { return static_cast<Derived &>(*this); }

  /// Operation filter.
  OpFilter opFilter{};

  /// Operand filter.
  OperandFilter operandFilter{};
};
} // namespace xb
} // namespace xblang

#endif
