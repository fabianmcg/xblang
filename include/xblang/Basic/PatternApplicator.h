//===- GenericPatternApplicator.h - Generic pattern applicator ---*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares a generic pattern applicator.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_BASIC_PATTERNAPPLICATOR_H
#define XBLANG_BASIC_PATTERNAPPLICATOR_H

#include "xblang/Basic/Pattern.h"

namespace xblang {
//===----------------------------------------------------------------------===//
// Pattern applicator
//===----------------------------------------------------------------------===//
/// This class manages the application of a group of rewrite patterns, with a
/// user-provided cost model.
class GenericPatternApplicator {
public:
  /// The cost model dynamically assigns a PatternBenefit to a particular
  /// pattern
  using CostModel = function_ref<PatternBenefit(const mlir::Pattern &)>;

  explicit GenericPatternApplicator(const FrozenPatternSet &frozenPatterns)
      : frozenPatterns(frozenPatterns) {}

  ~GenericPatternApplicator() = default;

  /// Attempt to match and rewrite the given op with any pattern, allowing a
  /// predicate to decide if a pattern can be applied or not, and hooks for if
  /// the pattern match was a success or failure.
  const GenericPattern *
  getPattern(Operation *op,
             function_ref<bool(const mlir::Pattern &)> canApply = {},
             function_ref<void(const mlir::Pattern &)> onFailure = {},
             function_ref<void(const mlir::Pattern &)> onSuccess = {});

  template <typename Pattern>
  const Pattern *
  getPattern(Operation *op,
             function_ref<bool(const mlir::Pattern &)> canApply = {},
             function_ref<void(const mlir::Pattern &)> onFailure = {},
             function_ref<void(const mlir::Pattern &)> onSuccess = {}) {
    return static_cast<const Pattern *>(
        getPattern(op, canApply, onFailure, onSuccess));
  }

  /// Apply a cost model to the patterns within this applicator.
  void applyCostModel(CostModel model);

  /// Apply the default cost model that solely uses the pattern's static
  /// benefit.
  void applyDefaultCostModel() {
    applyCostModel(
        [](const mlir::Pattern &pattern) { return pattern.getBenefit(); });
  }

private:
  /// List of fixed patterns.
  FrozenPatternSet frozenPatterns;
  /// The set of patterns to match for each operation, stable sorted by benefit.
  DenseMap<mlir::OperationName, SmallVector<const GenericPattern *, 2>>
      patterns;
  /// The set of patterns that may match against any operation type, stable
  /// sorted by benefit.
  SmallVector<const GenericPattern *, 1> anyOpPatterns;
};
} // namespace xblang

#endif // XBLANG_BASIC_PATTERNAPPLICATOR_H
