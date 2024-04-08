#ifndef XBLANG_DIALECT_XBLANG_ANALYSIS_SCOPEANALYSIS_H
#define XBLANG_DIALECT_XBLANG_ANALYSIS_SCOPEANALYSIS_H

#include "xblang/Dialect/XBLang/Analysis/Utils.h"

#include "mlir/IR/Value.h"
#include "llvm/ADT/SetVector.h"

namespace xblang {
namespace xb {
struct ScopeOpUses : public Value::use_range {
  using Base = Value::use_range;
  using Base::Base;

  ScopeOpUses() : Base(nullptr, nullptr) {}

  /// Check if it's valid.
  operator bool() const { return !empty(); }

  /// Compares 2 ranges.
  bool operator==(const ScopeOpUses &range) const {
    return begin() == range.begin();
  }

  /// Returns the value being used.
  Value getValue() { return begin().getOperand()->get(); }
};

class ScopedValueUseAnalysis : public AnalysisWalkBase {
public:
  using ScopeInfo = DenseMap<Operation *, ScopeOpUses>;

  ScopedValueUseAnalysis(Operation *topOp, OpFilter opFilter,
                         OperandFilter operandFilter);

  const ScopeInfo *getUses(Operation *limit) const {
    auto it = opUses.find(limit);
    if (it != opUses.end())
      return &(it->second);
    return nullptr;
  }

private:
  /// Cache the results.
  DenseMap<Operation *, ScopeInfo> opUses;
};

class SimpleScopedValueUseAnalysis : public AnalysisWalkBase {
public:
  using ValueUses = llvm::SmallSetVector<Value, 12>;
  using ValueUsers = llvm::SmallSetVector<Operation *, 12>;
  using UsesMap = DenseMap<Operation *, ValueUses>;
  using UsersMap = DenseMap<Value, ValueUsers>;

  SimpleScopedValueUseAnalysis(Operation *topOp, OpFilter opFilter,
                               OperandFilter operandFilter);

  /// Return all above defined uses inside the limit operation.
  const ValueUses *getUses(Operation *limit) const {
    auto it = valueUses.find(limit);
    if (it != valueUses.end())
      return &(it->second);
    return nullptr;
  }

  /// Return all operations using a value outside their scope.
  const ValueUsers *getUsers(Value value) const {
    auto it = valueUsers.find(value);
    if (it != valueUsers.end())
      return &(it->second);
    return nullptr;
  }

  /// Returns the uses map.
  const UsesMap &getUses() const { return valueUses; }

  /// Returns the users map.
  const UsersMap &getUsers() const { return valueUsers; }

private:
  // Cache the results.
  UsesMap valueUses;
  UsersMap valueUsers;
};
} // namespace xb
} // namespace xblang

#endif
