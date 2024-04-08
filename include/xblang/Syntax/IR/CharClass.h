//===- CharClass.h - Lex character classes -----------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares regex character classes.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_SYNTAX_IR_CHARCLASS_H
#define XBLANG_SYNTAX_IR_CHARCLASS_H

#include "llvm/Support/raw_ostream.h"
#include <set>

namespace xblang {
class SourceState;

namespace syntaxgen {
class CharClass;

/// Class representing character ranges.
class CharRange {
public:
  friend class CharClass;

  CharRange(uint32_t c) : lowerBound(c), upperBound(c) {}

  CharRange(uint32_t l, uint32_t u) : lowerBound(l), upperBound(u) {}

  /// Returns whether the ranges i valid or not.
  bool isValid() const { return lowerBound <= upperBound; }

  /// Returns whether the range represents a single character.
  bool isChar() const { return lowerBound == upperBound; }

  /// Returns whether the range represents a non-trivial range.
  bool isRange() const { return lowerBound < upperBound; }

  /// Compares two character ranges.
  bool operator<(const CharRange &range) const {
    return (lowerBound < range.lowerBound) ||
           ((lowerBound == range.lowerBound) &&
            (upperBound < range.upperBound));
  }

  /// Returns the lower bound of the range.
  uint32_t getLower() const { return lowerBound; }

  /// Returns the upper bound of the range.
  uint32_t getUpper() const { return upperBound; }

private:
  /// Lower bound of the range.
  uint32_t lowerBound;
  /// Upper bound of the range.
  uint32_t upperBound;
};

/// Class representing a character class.
class CharClass {
public:
  CharClass() = default;

  /// Inserts a single to the character class.
  void insert(uint32_t c) { charRanges.insert(CharRange(c)); }

  /// Inserts a char range to the character classs.
  void insert(uint32_t l, uint32_t u) { charRanges.insert(CharRange(l, u)); }

  /// Computes the USR representation.
  void computeUSR();

  /// Returns the USR.
  llvm::StringRef getUSR() const { return usr; }

  /// Returns the character ranges.
  const std::set<CharRange> &getRanges() const { return charRanges; }

  /// Compares two char classes for equality.
  bool operator==(const CharClass &) const;

  /// Returns a string representation.
  std::string toString() const;

  /// Creates a char class from a string. In case of failure it returns
  /// `std::nullopt`.
  static std::optional<CharClass> fromString(SourceState &str);

private:
  std::string usr;
  std::set<CharRange> charRanges;
};

/// Hashes a character class
llvm::hash_code hash_value(const CharClass &value);
} // namespace syntaxgen
} // namespace xblang

#endif // XBLANG_SYNTAX_IR_CHARCLASS_H
