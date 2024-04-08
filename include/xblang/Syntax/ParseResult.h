//===- ParseResult.cpp - Parse status ----------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines classes for holding parsing statuses and results.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_SYNTAX_PARSERESULT_H
#define XBLANG_SYNTAX_PARSERESULT_H

#include "xblang/Basic/SourceLocation.h"
#include <utility>

namespace xblang {
namespace syntax {
//===----------------------------------------------------------------------===//
// PureParsingStatus
//===----------------------------------------------------------------------===//
/// Class for representing a pure parsing status.
class PureParsingStatus {
public:
  typedef enum {
    Fatal = -2,
    Error = -1,
    Empty = 0,
    Success = 1,
  } Status;

  PureParsingStatus(const PureParsingStatus &) = default;
  PureParsingStatus &operator=(const PureParsingStatus &) = default;
  PureParsingStatus(PureParsingStatus &&) = default;
  PureParsingStatus &operator=(PureParsingStatus &&) = default;

  PureParsingStatus(Status status = Empty) : status(status) {}

  PureParsingStatus &operator=(Status status) {
    this->status = status;
    return *this;
  }

  /// Returns true on error.
  operator bool() const { return isAnError(); }

  /// Returns whether status is empty.
  bool isEmpty() const { return status == Empty; }

  /// Returns whether status is success.
  bool isSuccess() const { return status == Success; }

  /// Returns whether status is an error.
  bool isError() const { return status == Error; }

  /// Returns whether status is a fatal error.
  bool isFatal() const { return status == Fatal; }

  /// Returns whether status is any error.
  bool isAnError() const { return isError() || isFatal(); }

  /// Returns the status.
  Status getStatus() const { return status; }

protected:
  /// Parsing status.
  Status status;
};

//===----------------------------------------------------------------------===//
// ParsingStatus
//===----------------------------------------------------------------------===//
/// Class for representing a parsing status.
class ParsingStatus : public PureParsingStatus {
public:
  using PureParsingStatus::operator bool;
  using PureParsingStatus::operator=;
  ParsingStatus(const ParsingStatus &) = default;
  virtual ~ParsingStatus() = default;
  ParsingStatus &operator=(const ParsingStatus &) = default;
  ParsingStatus(ParsingStatus &&) = default;
  ParsingStatus &operator=(ParsingStatus &&) = default;

  ParsingStatus(Status status = ParsingStatus::Empty,
                SourceLocation location = {})
      : PureParsingStatus(status), location(location) {}

  ParsingStatus &operator=(const SourceLocation &location) {
    this->location = location;
    return *this;
  }

  /// Creates a new empty parsing status.
  static ParsingStatus empty(const SourceLocation &loc = {}) {
    return ParsingStatus(ParsingStatus::Empty, loc);
  }

  /// Creates a new success parsing status.
  static ParsingStatus success(const SourceLocation &loc = {}) {
    return ParsingStatus(ParsingStatus::Success, loc);
  }

  /// Creates a new error parsing status.
  static ParsingStatus error(const SourceLocation &loc = {}) {
    return ParsingStatus(ParsingStatus::Error, loc);
  }

  /// Creates a new fatal parsing status.
  static ParsingStatus fatal(const SourceLocation &loc = {}) {
    return ParsingStatus(ParsingStatus::Fatal, loc);
  }

  /// Returns the location hold by the status.
  SourceLocation getLoc() const { return location; }

protected:
  /// Location of the status.
  SourceLocation location;
};

//===----------------------------------------------------------------------===//
// ParseResult
//===----------------------------------------------------------------------===//
/// Class for returning from parsing methods.
template <typename T>
class ParseResult : public ParsingStatus {
public:
  using value_type = T;
  using ParsingStatus::operator=;
  using ParsingStatus::operator bool;
  ParseResult() = default;

  ParseResult(const ParsingStatus &status) : ParsingStatus(status) {}

  ParseResult(const ParseResult &other)
      : ParsingStatus(other), value(other.value) {}

  ParseResult(ParseResult &&other)
      : ParsingStatus(other), value(std::exchange(other.value, value_type{})) {}

  ParseResult &operator=(ParseResult &&other) {
    value = std::exchange(other.value, value_type{});
    static_cast<ParsingStatus &>(*this) = other;
    return *this;
  }

  template <typename V>
  ParseResult &operator=(V &&val) {
    value = std::forward<V>(val);
    return *this;
  }

  operator value_type() const { return value; }

  /// Provides access to the held value.
  value_type *operator->() { return &value; }

  const value_type *operator->() const { return &value; }

  /// Returns the held value.
  value_type &get() { return value; }

  value_type get() const { return value; }

  /// Creates a new parsing result.
  static ParseResult make(value_type &&value,
                          const ParsingStatus &status = {}) {
    return ParseResult(status.getStatus(), std::forward<value_type>(value),
                       status.getLoc());
  }

  /// Creates a new empty parsing result.
  static ParseResult empty(const SourceLocation &loc = {}) {
    return ParsingStatus(ParsingStatus::Empty, loc);
  }

  /// Creates a new success parsing result.
  static ParseResult success(value_type &&value,
                             const SourceLocation &loc = {}) {
    return ParseResult(ParsingStatus::Success, std::forward<value_type>(value),
                       loc);
  }

  /// Creates a new error parsing result.
  static ParseResult error(const SourceLocation &loc = {}) {
    return ParsingStatus(ParsingStatus::Error, loc);
  }

  /// Creates a new fatal parsing result.
  static ParseResult fatal(const SourceLocation &loc = {}) {
    return ParsingStatus(ParsingStatus::Fatal, loc);
  }

protected:
  ParseResult(Status status, value_type &&value,
              const SourceLocation &location = {})
      : ParsingStatus(status, location),
        value(std::forward<value_type>(value)) {}

  ParseResult(Status status, const value_type &value,
              const SourceLocation &location = {})
      : ParsingStatus(status, location), value(value) {}

private:
  /// Value being stored.
  value_type value{};
};

//===----------------------------------------------------------------------===//
// ParseResult<VoidResult>
//===----------------------------------------------------------------------===//
struct VoidResult {};

template <>
class ParseResult<VoidResult> : public ParsingStatus {
public:
  using ParsingStatus::operator=;
  using ParsingStatus::operator bool;
  ParseResult() = default;

  ParseResult(const ParsingStatus &status) : ParsingStatus(status) {}

  /// Creates a new empty parsing status.
  static ParseResult empty(const SourceLocation &loc = {}) {
    return ParsingStatus(ParsingStatus::Empty, loc);
  }

  /// Creates a new success parsing status.
  static ParseResult success(const SourceLocation &loc = {}) {
    return ParsingStatus(ParsingStatus::Success, loc);
  }

  /// Creates a new error parsing status.
  static ParseResult error(const SourceLocation &loc = {}) {
    return ParsingStatus(ParsingStatus::Error, loc);
  }

  /// Creates a new fatal parsing status.
  static ParseResult fatal(const SourceLocation &loc = {}) {
    return ParsingStatus(ParsingStatus::Fatal, loc);
  }
};

//===----------------------------------------------------------------------===//
// ParseCachedResult
//===----------------------------------------------------------------------===//
template <typename T>
class ParseCachedResult : public ParseResult<T> {
private:
  using base = ParseResult<T>;
  using value_type = T;
  using Status = PureParsingStatus::Status;

public:
  ParseCachedResult(Status status, value_type &&value,
                    const SourceLocation &location = {}, const char *end = {})
      : base(status, std::forward<value_type>(value), location), end(end) {}

  ParseCachedResult(Status status, const value_type &value,
                    const SourceLocation &location = {}, const char *end = {})
      : base(status, value, location), end(end) {}

  virtual ~ParseCachedResult() = default;
  using base::operator=;

  const char *getEnd() const { return end; }

private:
  const char *end{};
};

//===----------------------------------------------------------------------===//
// DynParsingCombinatorTraits
//===----------------------------------------------------------------------===//
template <typename T>
struct DynParsingCombinatorTraits {
  using ParsingResult = ParseResult<VoidResult>;
};

template <typename T>
using combinator_result = typename DynParsingCombinatorTraits<T>::ParsingResult;
} // namespace syntax
} // namespace xblang

#endif
