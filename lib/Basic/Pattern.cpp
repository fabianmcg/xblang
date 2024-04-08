//===- Pattern.cpp - XB patterns ---------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines XB generic patterns and related classes.
//
//===----------------------------------------------------------------------===//

#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "xblang/Basic/Context.h"
#include "xblang/Basic/PatternApplicator.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "xb-pattern-application"

using namespace xblang;

//===----------------------------------------------------------------------===//
// GenericPatternSet
//===----------------------------------------------------------------------===//

MLIRContext *GenericPatternSet::getMLIRContext() const {
  return getContext()->getMLIRContext();
}

//===----------------------------------------------------------------------===//
// GenericPatternApplicator
//===----------------------------------------------------------------------===//

FrozenPatternSet::FrozenPatternSet(const GenericPatternSet &patterns)
    : context(patterns.getContext()) {
  std::vector<mlir::RegisteredOperationName> opInfos;
  impl.reset(new Impl);
  auto addToOpsWhen =
      [&](const GenericPattern *pattern,
          function_ref<bool(mlir::RegisteredOperationName)> callbackFn) {
        if (opInfos.empty())
          opInfos = pattern->getContext()->getRegisteredOperations();
        for (mlir::RegisteredOperationName info : opInfos)
          if (callbackFn(info))
            impl->opPatterns[info].push_back(pattern);
      };
  for (const auto &pat : patterns.patternSet) {
    if (std::optional<mlir::OperationName> rootName = pat->getRootKind()) {
      impl->opPatterns[*rootName].push_back(pat.get());
      continue;
    }
    if (std::optional<mlir::TypeID> interfaceID = pat->getRootInterfaceID()) {
      addToOpsWhen(pat.get(), [&](mlir::RegisteredOperationName info) {
        return info.hasInterface(*interfaceID);
      });
      continue;
    }
    if (std::optional<mlir::TypeID> traitID = pat->getRootTraitID()) {
      addToOpsWhen(pat.get(), [&](mlir::RegisteredOperationName info) {
        return info.hasTrait(*traitID);
      });
      continue;
    }
    impl->anyPatterns.push_back(pat.get());
  }
}

//===----------------------------------------------------------------------===//
// GenericPatternApplicator
//===----------------------------------------------------------------------===//

#ifndef NDEBUG
/// Log a message for a pattern that is impossible to match.
static void logImpossibleToMatch(const mlir::Pattern &pattern) {
  llvm::dbgs() << "Ignoring pattern '" << pattern.getRootKind()
               << "' because it is impossible to match or cannot lead "
                  "to legal IR (by cost model)\n";
}
#endif

void GenericPatternApplicator::applyCostModel(CostModel model) {
  // Copy over the patterns so that we can sort by benefit based on the cost
  // model. Patterns that are already impossible to match are ignored.
  patterns.clear();
  for (const auto &it : frozenPatterns.getOpPatterns()) {
    for (const GenericPattern *pattern : it.second) {
      if (pattern->getBenefit().isImpossibleToMatch())
        LLVM_DEBUG(logImpossibleToMatch(*pattern));
      else
        patterns[it.first].push_back(pattern);
    }
  }
  anyOpPatterns.clear();
  for (const GenericPattern *pattern : frozenPatterns.getMatchAnyOpPatterns()) {
    if (pattern->getBenefit().isImpossibleToMatch())
      LLVM_DEBUG(logImpossibleToMatch(*pattern));
    else
      anyOpPatterns.push_back(pattern);
  }

  // Sort the patterns using the provided cost model.
  llvm::SmallDenseMap<const GenericPattern *, PatternBenefit> benefits;
  auto cmp = [&benefits](const GenericPattern *lhs, const GenericPattern *rhs) {
    return benefits[lhs] > benefits[rhs];
  };
  auto processPatternList = [&](SmallVectorImpl<const GenericPattern *> &list) {
    // Special case for one pattern in the list, which is the most common case.
    if (list.size() == 1) {
      if (model(*list.front()).isImpossibleToMatch()) {
        LLVM_DEBUG(logImpossibleToMatch(*list.front()));
        list.clear();
      }
      return;
    }

    // Collect the dynamic benefits for the current pattern list.
    benefits.clear();
    for (const GenericPattern *pat : list)
      benefits.try_emplace(pat, model(*pat));

    // Sort patterns with highest benefit first, and remove those that are
    // impossible to match.
    std::stable_sort(list.begin(), list.end(), cmp);
    while (!list.empty() && benefits[list.back()].isImpossibleToMatch()) {
      LLVM_DEBUG(logImpossibleToMatch(*list.back()));
      list.pop_back();
    }
  };
  for (auto &it : patterns)
    processPatternList(it.second);
  processPatternList(anyOpPatterns);
}

const GenericPattern *GenericPatternApplicator::getPattern(
    Operation *op, function_ref<bool(const mlir::Pattern &)> canApply,
    function_ref<void(const mlir::Pattern &)> onFailure,
    function_ref<void(const mlir::Pattern &)> onSuccess) {
  MutableArrayRef<const GenericPattern *> opPatterns;
  auto patternIt = patterns.find(op->getName());
  if (patternIt != patterns.end())
    opPatterns = patternIt->second;

  unsigned opIt = 0, opE = opPatterns.size();
  unsigned anyIt = 0, anyE = anyOpPatterns.size();
  const GenericPattern *bestPattern;
  do {
    LogicalResult result = failure();
    // Find the next pattern with the highest benefit.
    bestPattern = nullptr;
    unsigned *bestPatternIt = &opIt;

    /// Operation specific patterns.
    if (opIt < opE)
      bestPattern = opPatterns[opIt];
    /// Operation agnostic patterns.
    if (anyIt < anyE &&
        (!bestPattern ||
         bestPattern->getBenefit() < anyOpPatterns[anyIt]->getBenefit())) {
      bestPatternIt = &anyIt;
      bestPattern = anyOpPatterns[anyIt];
    }

    if (!bestPattern)
      break;

    // Update the pattern iterator on failure so that this pattern isn't
    // attempted again.
    ++(*bestPatternIt);

    // Check that the pattern can be applied.
    if (canApply && !canApply(*bestPattern))
      continue;

    // Try to match this pattern.
    LLVM_DEBUG(llvm::dbgs() << "Trying to match \""
                            << bestPattern->getDebugName() << "\"\n");

    result = bestPattern->match(op);

    LLVM_DEBUG(llvm::dbgs() << "\"" << bestPattern->getDebugName()
                            << "\" result " << succeeded(result) << "\n");

    // Process the result of the pattern application.
    if (succeeded(result)) {
      if (onSuccess)
        onSuccess(*bestPattern);
      break;
    }

    if (onFailure)
      onFailure(*bestPattern);
  } while (true);
  return bestPattern;
}
