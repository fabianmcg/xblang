//===- Pattern.h - XB Patterns -----------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares XB generic patterns.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_BASIC_PATTERN_H
#define XBLANG_BASIC_PATTERN_H

#include "mlir/IR/PatternMatch.h"
#include "xblang/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"

namespace xblang {
class XBContext;

//===----------------------------------------------------------------------===//
// GenericPattern
//===----------------------------------------------------------------------===//
/// GenericPattern is the common base class for all XB patterns.
class GenericPattern : public mlir::Pattern {
public:
  virtual ~GenericPattern() = default;

  /// Attempt to match against code rooted at the specified operation,
  /// which is the same operation code as getRootKind().
  virtual LogicalResult match(Operation *op) const { return success(); }

  /// This method provides a convenient interface for creating and initializing
  /// derived rewrite patterns of the given type `T`.
  template <typename T, typename... Args>
  static std::unique_ptr<T> create(Args &&...args) {
    std::unique_ptr<T> pattern =
        std::make_unique<T>(std::forward<Args>(args)...);
    initializePattern<T>(*pattern);

    // Set a default debug name if one wasn't provided.
    if (pattern->getDebugName().empty())
      pattern->setDebugName(llvm::getTypeName<T>());
    return pattern;
  }

protected:
  /// Inherit the base constructors from `Pattern`.
  using Pattern::Pattern;

private:
  /// Trait to check if T provides a `getOperationName` method.
  template <typename T, typename... Args>
  using has_initialize = decltype(std::declval<T>().initialize());
  template <typename T>
  using detect_has_initialize = llvm::is_detected<has_initialize, T>;

  /// Initialize the derived pattern by calling its `initialize` method.
  template <typename T>
  static std::enable_if_t<detect_has_initialize<T>::value>
  initializePattern(T &pattern) {
    pattern.initialize();
  }

  /// Empty derived pattern initializer for patterns that do not have an
  /// initialize method.
  template <typename T>
  static std::enable_if_t<!detect_has_initialize<T>::value>
  initializePattern(T &) {}
};

//===----------------------------------------------------------------------===//
// Generic pattern set
//===----------------------------------------------------------------------===//
class FrozenPatternSet;

/// A class for holding generic patterns.
class GenericPatternSet {
public:
  GenericPatternSet(XBContext *context) : context(context) {}

  GenericPatternSet(GenericPatternSet &&) = default;
  GenericPatternSet(const GenericPatternSet &) = delete;

  /// Add a pattern to the set.
  template <typename T, typename... Args>
  void addPattern(Args &&...args) {
    patternSet.emplace_back(
        GenericPattern::create<T>(std::forward<Args>(args)...));
  }

  /// Add a pattern to the set.
  template <typename... T, typename Arg, typename... Args>
  void add(Arg &&arg, Args &&...args) {
    (addPattern<T>(std::forward<Arg>(arg), std::forward<Args>(args)...), ...);
  }

  template <typename... T>
  void add() {
    (addPattern<T>(), ...);
  }

  /// Returns the XB context.
  XBContext *getContext() const { return context; }

  /// Returns the XB context.
  MLIRContext *getMLIRContext() const;

private:
  friend class FrozenPatternSet;
  XBContext *context;
  SmallVector<std::unique_ptr<GenericPattern>> patternSet;
};

/// Class for holding a fixed set of patterns.
class FrozenPatternSet {
public:
  using PatternList = SmallVector<const GenericPattern *, 0>;
  using PatternMap = DenseMap<mlir::OperationName, PatternList>;
  using FrozenSet = ArrayRef<const GenericPattern *>;
  FrozenPatternSet() = delete;
  explicit FrozenPatternSet(const GenericPatternSet &patterns);
  FrozenPatternSet(FrozenPatternSet &&) = default;
  FrozenPatternSet(const FrozenPatternSet &) = default;

  /// Return the op specific native patterns held by this list.
  const PatternMap &getOpPatterns() const { return impl->opPatterns; }

  /// Return the "match any" native patterns held by this list.
  FrozenSet getMatchAnyOpPatterns() const { return impl->anyPatterns; }

  /// Returns the XB context.
  XBContext *getContext() const { return context; }

  /// Returns the XB context.
  MLIRContext *getMLIRContext() const;

private:
  struct Impl {
    PatternMap opPatterns{};
    PatternList anyPatterns{};
  };

  std::shared_ptr<Impl> impl;
  XBContext *context{};
};
} // namespace xblang

#endif // XBLANG_BASIC_PATTERN_H
