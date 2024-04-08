//===- SourceState.h - Source state -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the class `SourceState`.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_BASIC_SOURCESTATE_H
#define XBLANG_BASIC_SOURCESTATE_H

#include "xblang/Basic/SourceLocation.h"
#include "llvm/ADT/StringRef.h"
#include <cassert>

namespace xblang {
/// This class represents a source string with a location.
class SourceString {
protected:
  SourceString(const char *bufferBegin, const char *bufferEnd, int line = 1,
               int column = 1)
      : buffer(bufferBegin), bufferEnd(bufferEnd), line(line), column(column) {
    if (!isValid()) {
      this->bufferEnd = buffer = nullptr;
      this->line = this->column = -1;
    }
  }

public:
  SourceString() = default;
  SourceString(SourceString &&) = default;
  SourceString(const SourceString &) = default;
  SourceString &operator=(SourceString &&) = default;
  SourceString &operator=(const SourceString &) = default;

  /// Create a lexer state from a StringRef.
  SourceString(llvm::StringRef buffer, int line = 1, int column = 1)
      : SourceString(buffer.data(), buffer.data() + buffer.size(), line,
                     column) {}

  /// Returns whether two states are the same.
  inline bool operator==(const SourceString &other) const {
    return buffer == other.buffer;
  }

  /// Returns whether the state and the location are the same.
  inline bool operator==(const SourceLocation &other) const {
    return buffer == other.loc.getPointer();
  }

  /// Returns the current char.
  inline uint32_t operator*() const {
    assert(buffer && "Invalid buffer.");
    if (buffer < bufferEnd)
      return *buffer;
    return 0;
  }

  /// Returns the char at location buffer + i
  inline uint32_t operator[](int i) const {
    assert(buffer && "Invalid buffer.");
    return buffer[i];
  }

  /// Returns the char at location buffer + i
  inline uint32_t at(size_t i) const {
    assert(buffer && "Invalid buffer.");
    auto ptr = buffer + i;
    if (ptr < bufferEnd)
      return *ptr;
    return 0;
  }

  /// Returns the current char.
  inline uint32_t get() const { return **this; }

  /// Returns a pointer to the buffer.
  const char *begin() const { return buffer; }

  /// Returns a pointer to the end of the buffer.
  const char *end() const { return bufferEnd; }

  /// Returns the column number.
  inline int getColumn() const { return column; }

  /// Returns the line number.
  inline int getLine() const { return line; }

  /// Returns the size of the buffer.
  inline size_t size() const {
    assert((bufferEnd - buffer) >= 0);
    return bufferEnd - buffer;
  }

  /// Returns whether the buffer is valid.
  static inline bool isValid(const char *bufferBegin, const char *bufferEnd) {
    return bufferBegin && (bufferBegin <= bufferEnd);
  }

  /// Returns whether the buffer is valid.
  inline bool isValid() const { return isValid(buffer, bufferEnd); }

  /// Returns the current source location of the buffer.
  inline SourceLocation getLoc() const {
    return SourceLocation(line, column, buffer);
  }

  /// Returns the buffer as StringRef
  inline llvm::StringRef str() const {
    if (isValid())
      return llvm::StringRef(buffer, size());
    return {};
  }

  operator llvm::StringRef() const { return str(); }

protected:
  /// Current location of the buffer.
  const char *buffer{};
  /// End of the buffer.
  const char *bufferEnd{};
  /// Current line.
  int line{-1};
  /// Current column.
  int column{-1};
};

/// Class holding the internal buffer state of a source.
class SourceState : public SourceString {
public:
  using SourceString::SourceString;
  using SourceString::operator==;

  /// Returns whether two states are the same.
  inline bool operator==(const SourceState &other) const {
    return buffer == other.buffer;
  }

  /// Advances the buffer and returns the new buffer pointer.
  inline const char *operator++() { return advance().buffer; }

  /// Advances the buffer and returns the old buffer pointer.
  inline const char *operator++(int) {
    auto tmp = buffer;
    advance();
    return tmp;
  }

  /// Returns whether the buffer is at the end of file.
  inline bool isEndOfFile() const { return (buffer >= bufferEnd); }

  /// Restores the lexer state from a source location.
  inline void restore(const SourceLocation &location) {
    const char *buf = location.getLoc();
    assert(isValid(buf, bufferEnd) && "Invalid buffer.");
    buffer = buf;
    line = location.line;
    column = location.column;
  }

  /// Restores the lexer state from a source location and end buffer pointer.
  inline void restore(const SourceLocation &location, const char *bufferEnd) {
    assert(isValid(location.getLoc(), bufferEnd));
    buffer = location.getLoc();
    line = location.line;
    column = location.column;
    this->bufferEnd = bufferEnd;
  }

  /// Advances the buffer by a single character.
  inline SourceState &advance() {
    assert(buffer && buffer < bufferEnd && "Invalid buffer.");
    auto curentChar = *(buffer++);
    if (curentChar == '\n' || curentChar == '\r') {
      ++line;
      column = 1;
    } else
      ++column;
    return *this;
  }
};
} // namespace xblang

#endif // XBLANG_BASIC_SOURCESTATE_H
