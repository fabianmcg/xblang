//===- ParsingCombinators.h - Common parsing combinators ---------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares common parsing combinators.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_SYNTAX_PARSINGCOMBINATORS_H
#define XBLANG_SYNTAX_PARSINGCOMBINATORS_H

#include "xblang/Syntax/ParseResult.h"

namespace mlir {
class Operation;
class Value;
} // namespace mlir

namespace xblang {
namespace xlg {
struct Decl;
struct Stmt;
struct Expr;
struct Type;
} // namespace xlg

namespace syntax {
template <>
struct DynParsingCombinatorTraits<xlg::Attr> {
  using ParsingResult = ParseResult<::mlir::Attribute>;
};

template <>
struct DynParsingCombinatorTraits<xlg::Decl> {
  using ParsingResult = ParseResult<::mlir::Operation *>;
};

template <>
struct DynParsingCombinatorTraits<xlg::Expr> {
  using ParsingResult = ParseResult<mlir::Value>;
};

template <>
struct DynParsingCombinatorTraits<xlg::Stmt> {
  using ParsingResult = ParseResult<::mlir::Operation *>;
};

template <>
struct DynParsingCombinatorTraits<xlg::Type> {
  using ParsingResult = ParseResult<mlir::Value>;
};
} // namespace syntax
} // namespace xblang

#endif // XBLANG_SYNTAX_PARSINGCOMBINATORS_H
