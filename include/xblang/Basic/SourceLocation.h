//===- SourceLocation.h - Source location ------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines classes related to source code locations.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_BASIC_SOURCELOCATION_H
#define XBLANG_BASIC_SOURCELOCATION_H

#include "llvm/Support/SMLoc.h"
#include <string>

namespace xblang {
//===----------------------------------------------------------------------===//
// SourceLocation
//===----------------------------------------------------------------------===//
/// Class for representing source code locations.
struct SourceLocation {
  SourceLocation() = default;
  ~SourceLocation() = default;
  SourceLocation(SourceLocation &&) = default;
  SourceLocation(const SourceLocation &) = default;
  SourceLocation &operator=(SourceLocation &&) = default;
  SourceLocation &operator=(const SourceLocation &) = default;

  /// Construct a location from a line, a column and a pointer to the buffer.
  SourceLocation(int l, int c, const char *ptr)
      : loc(llvm::SMLoc::getFromPointer(ptr)), line(l), column(c) {}

  /// Returns true if the location points to a valid location.
  bool valid() const { return line > 0 && column > 0 && loc.getPointer(); }

  /// 3-way comparison of the buffer pointers.
  constexpr auto operator<=>(const SourceLocation &other) const {
    return loc.getPointer() <=> other.loc.getPointer();
  }

  /// Returns true if the two source locations are equal.
  bool operator==(const SourceLocation &other) const {
    return (*this <=> other) == 0;
  }

  /// Returns the char pointer to the location.
  const char *getLoc() const { return loc.getPointer(); }

  /// Converts the location to a string representation containing the line and
  /// column.
  std::string toString() const {
    return "[" + std::to_string(line) + ":" + std::to_string(column) + "]";
  }

  /// Pointer to the location in the buffer.
  llvm::SMLoc loc{};
  /// Line number.
  int line{-1};
  /// Column number.
  int column{-1};
};

//===----------------------------------------------------------------------===//
// SourceRange
//===----------------------------------------------------------------------===//
/// Class for representing source ranges.
struct SourceRange {
  ~SourceRange() = default;
  SourceRange(SourceRange &&) = default;
  SourceRange(const SourceRange &) = default;
  SourceRange &operator=(SourceRange &&) = default;
  SourceRange &operator=(const SourceRange &) = default;

  /// Creates a source range from two locations.
  SourceRange(const SourceLocation &begin = {}, const SourceLocation &end = {})
      : begin(begin), end(end) {}

  /// Returns true if both ranges are valid and begin <= end, false otherwise.
  bool valid() const { return begin.valid() && end.valid() && (begin <= end); }

  /// Start of the source range.
  SourceLocation begin{};

  /// End of the source range.
  SourceLocation end{};
};
} // namespace xblang

#endif // XBLANG_BASIC_SOURCELOCATION_H
