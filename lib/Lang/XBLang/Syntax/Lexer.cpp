//===- Lexer.cpp - Defines the XBLang lexer ----------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the xblang lexer.
//
//===----------------------------------------------------------------------===//

#include "xblang/Lang/XBLang/Syntax/Lexer.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"

#include "xblang/Lang/XBLang/Syntax/XBLangLexer.cpp.inc"

namespace xblang {
xblang::BinaryOperator XBLangLexer::toBinaryOp(TokenID kind) {
  switch (kind) {
  case XBLangLexer::Equal:
    return BinaryOperator::Assign;
  case XBLangLexer::CompoundPlus:
    return BinaryOperator::CompoundAdd;
  case XBLangLexer::CompoundMinus:
    return BinaryOperator::CompoundSub;
  case XBLangLexer::CompoundMultiply:
    return BinaryOperator::CompoundMul;
  case XBLangLexer::CompoundDivide:
    return BinaryOperator::CompoundDiv;
  case XBLangLexer::CompoundMod:
    return BinaryOperator::CompoundMod;
  case XBLangLexer::CompoundAnd:
    return BinaryOperator::CompoundAnd;
  case XBLangLexer::CompoundOr:
    return BinaryOperator::CompoundOr;
  case XBLangLexer::CompoundBAnd:
    return BinaryOperator::CompoundBinaryAnd;
  case XBLangLexer::CompoundBOr:
    return BinaryOperator::CompoundBinaryOr;
  case XBLangLexer::CompoundBXor:
    return BinaryOperator::CompoundBinaryXor;
  case XBLangLexer::CompoundLShift:
    return BinaryOperator::CompoundLShift;
  case XBLangLexer::CompoundRShift:
    return BinaryOperator::CompoundRShift;
  case XBLangLexer::Question:
    return BinaryOperator::Ternary;
  case XBLangLexer::Or:
    return BinaryOperator::Or;
  case XBLangLexer::And:
    return BinaryOperator::And;
  case XBLangLexer::Hat:
    return BinaryOperator::BinaryXor;
  case XBLangLexer::VerBar:
    return BinaryOperator::BinaryOr;
  case XBLangLexer::Ampersand:
    return BinaryOperator::BinaryAnd;
  case XBLangLexer::Less:
    return BinaryOperator::Less;
  case XBLangLexer::Greater:
    return BinaryOperator::Greater;
  case XBLangLexer::LEq:
    return BinaryOperator::LEQ;
  case XBLangLexer::GEq:
    return BinaryOperator::GEQ;
  case XBLangLexer::Equality:
    return BinaryOperator::Equal;
  case XBLangLexer::NEq:
    return BinaryOperator::NEQ;
  case XBLangLexer::Spaceship:
    return BinaryOperator::Spaceship;
  case XBLangLexer::LShift:
    return BinaryOperator::LShift;
  case XBLangLexer::RShift:
    return BinaryOperator::RShift;
  case XBLangLexer::Plus:
    return BinaryOperator::Add;
  case XBLangLexer::Dash:
    return BinaryOperator::Sub;
  case XBLangLexer::Asterisk:
    return BinaryOperator::Mul;
  case XBLangLexer::Slash:
    return BinaryOperator::Div;
  case XBLangLexer::Percent:
    return BinaryOperator::Mod;
  case XBLangLexer::Dot:
    return BinaryOperator::Dot;
  default:
    return BinaryOperator::firstBinOp;
  }
}

xblang::UnaryOperator XBLangLexer::toUnaryOp(TokenID kind) {
  switch (kind) {
  case XBLangLexer::Ampersand:
    return UnaryOperator::Address;
  case XBLangLexer::Asterisk:
    return UnaryOperator::Dereference;
  case XBLangLexer::Plus:
    return UnaryOperator::Plus;
  case XBLangLexer::Dash:
    return UnaryOperator::Minus;
  case XBLangLexer::Exclamation:
    return UnaryOperator::Negation;
  case XBLangLexer::Increment:
    return UnaryOperator::Increment;
  case XBLangLexer::Decrement:
    return UnaryOperator::Decrement;
  default:
    return UnaryOperator::None;
  }
}

XBLangLexer::IntLiteralInfo XBLangLexer::getIntInfo(TokenID kind) {
  switch (kind) {
  // Decimals literals.
  case IntLiteral:
    return IntLiteralInfo({10, 0u, mlir::IntegerType::Signed});
  case IntLiterali8:
    return IntLiteralInfo({10, 8u, mlir::IntegerType::Signed});
  case IntLiterali16:
    return IntLiteralInfo({10, 16u, mlir::IntegerType::Signed});
  case IntLiterali32:
    return IntLiteralInfo({10, 32u, mlir::IntegerType::Signed});
  case IntLiterali64:
    return IntLiteralInfo({10, 64u, mlir::IntegerType::Signed});
  case IntLiteralu:
    return IntLiteralInfo({10, 0u, mlir::IntegerType::Unsigned});
  case IntLiteralu8:
    return IntLiteralInfo({10, 8u, mlir::IntegerType::Unsigned});
  case IntLiteralu16:
    return IntLiteralInfo({10, 16u, mlir::IntegerType::Unsigned});
  case IntLiteralu32:
    return IntLiteralInfo({10, 32u, mlir::IntegerType::Unsigned});
  case IntLiteralu64:
    return IntLiteralInfo({10, 64u, mlir::IntegerType::Unsigned});
    // Binary literals.
  case BinaryIntLiteral:
    return IntLiteralInfo({2, 0u, mlir::IntegerType::Signed});
  case BinaryIntLiterali8:
    return IntLiteralInfo({2, 8u, mlir::IntegerType::Signed});
  case BinaryIntLiterali16:
    return IntLiteralInfo({2, 16u, mlir::IntegerType::Signed});
  case BinaryIntLiterali32:
    return IntLiteralInfo({2, 32u, mlir::IntegerType::Signed});
  case BinaryIntLiterali64:
    return IntLiteralInfo({2, 64u, mlir::IntegerType::Signed});
  case BinaryIntLiteralu:
    return IntLiteralInfo({2, 0u, mlir::IntegerType::Unsigned});
  case BinaryIntLiteralu8:
    return IntLiteralInfo({2, 8u, mlir::IntegerType::Unsigned});
  case BinaryIntLiteralu16:
    return IntLiteralInfo({2, 16u, mlir::IntegerType::Unsigned});
  case BinaryIntLiteralu32:
    return IntLiteralInfo({2, 32u, mlir::IntegerType::Unsigned});
  case BinaryIntLiteralu64:
    return IntLiteralInfo({2, 64u, mlir::IntegerType::Unsigned});
    // Octal Literals.
  case OctalIntLiteral:
    return IntLiteralInfo({8, 0u, mlir::IntegerType::Signed});
  case OctalIntLiterali8:
    return IntLiteralInfo({8, 8u, mlir::IntegerType::Signed});
  case OctalIntLiterali16:
    return IntLiteralInfo({8, 16u, mlir::IntegerType::Signed});
  case OctalIntLiterali32:
    return IntLiteralInfo({8, 32u, mlir::IntegerType::Signed});
  case OctalIntLiterali64:
    return IntLiteralInfo({8, 64u, mlir::IntegerType::Signed});
  case OctalIntLiteralu:
    return IntLiteralInfo({8, 0u, mlir::IntegerType::Unsigned});
  case OctalIntLiteralu8:
    return IntLiteralInfo({8, 8u, mlir::IntegerType::Unsigned});
  case OctalIntLiteralu16:
    return IntLiteralInfo({8, 16u, mlir::IntegerType::Unsigned});
  case OctalIntLiteralu32:
    return IntLiteralInfo({8, 32u, mlir::IntegerType::Unsigned});
  case OctalIntLiteralu64:
    return IntLiteralInfo({8, 64u, mlir::IntegerType::Unsigned});
    // Hex literals.
  case HexIntLiteral:
    return IntLiteralInfo({16, 0u, mlir::IntegerType::Signed});
  case HexIntLiterali8:
    return IntLiteralInfo({16, 8u, mlir::IntegerType::Signed});
  case HexIntLiterali16:
    return IntLiteralInfo({16, 16u, mlir::IntegerType::Signed});
  case HexIntLiterali32:
    return IntLiteralInfo({16, 32u, mlir::IntegerType::Signed});
  case HexIntLiterali64:
    return IntLiteralInfo({16, 64u, mlir::IntegerType::Signed});
  case HexIntLiteralu:
    return IntLiteralInfo({16, 0u, mlir::IntegerType::Unsigned});
  case HexIntLiteralu8:
    return IntLiteralInfo({16, 8u, mlir::IntegerType::Unsigned});
  case HexIntLiteralu16:
    return IntLiteralInfo({16, 16u, mlir::IntegerType::Unsigned});
  case HexIntLiteralu32:
    return IntLiteralInfo({16, 32u, mlir::IntegerType::Unsigned});
  case HexIntLiteralu64:
    return IntLiteralInfo({16, 64u, mlir::IntegerType::Unsigned});
  default:
    break;
  }
  return IntLiteralInfo({-1, 0, mlir::IntegerType::Signless});
}

namespace {
mlir::LogicalResult lexEscape(SourceState &state,
                              llvm::SmallString<128> &literal,
                              const SMDiagnosticsEmitter &emitter) {
  bool consume = true;
  switch (state.at(1)) {
  case '\'':
    literal.push_back('\'');
    break;
  case '\"':
    literal.push_back('\"');
    break;
  case '\\':
    literal.push_back('\\');
    break;
  case 'a':
    literal.push_back('\a');
    break;
  case 'b':
    literal.push_back('\b');
    break;
  case 'f':
    literal.push_back('\f');
    break;
  case 'n':
    literal.push_back('\n');
    break;
  case 'r':
    literal.push_back('\r');
    break;
  case 't':
    literal.push_back('\t');
    break;
  case 'v':
    literal.push_back('\v');
    break;
  case 'u':
  case 'U': {
    state.advance();
    state.advance();
    if (*state != '+') {
      emitter.emitError(state.getLoc(),
                        "invalid unicode character: missing `+` char");
      return mlir::failure();
    }
    state.advance();
    auto begin = state.begin();
    while (isxdigit(*state)) {
      literal.push_back(llvm::hexDigitValue(*state));
      state.advance();
    }
    auto end = state.begin();
    if (begin == end) {
      emitter.emitError(state.getLoc(), "invalid unicode character: is empty");
      return mlir::failure();
    }
    uint32_t code = 0;
    if (llvm::StringRef(begin, end - begin).getAsInteger(16, code)) {
      emitter.emitError(state.getLoc(), "invalid unicode character for UTF-32");
      return mlir::failure();
    }
    break;
  }
  default:
    emitter.emitError(state.getLoc(), "invalid escaped character");
    return mlir::failure();
  }
  if (consume) {
    state.advance();
    state.advance();
  }
  return mlir::success();
}
} // namespace

XBLangLexer::TokenID XBLangLexer::parseString(SourceState &state,
                                              llvm::StringRef &spelling) const {
  uint32_t quote = state[-1];
  bool tripleQuote = false;
  bool finished = false;
  if (*state == quote && state.at(1) == quote) {
    tripleQuote = true;
    state.advance();
    state.advance();
  }
  bool escaped = false;
  const char *begin = state.begin();
  const char *end = state.begin();
  llvm::SmallString<128> literal;
  while (state.isValid() && !state.isEndOfFile()) {
    // Lex escape sequences if we are not in a triple quote.
    if (!tripleQuote && *state == '\\') {
      escaped = true;
      if (mlir::failed(lexEscape(state, literal, *this)))
        return Invalid;
      continue;
    }
    // Check for string termination.
    if (*state == quote) {
      auto tmp = state.begin();
      if (tripleQuote) {
        if (state.at(1) == quote && state.at(2) == quote) {
          state.advance();
          state.advance();
        } else {
          state.advance();
          continue;
        }
      }
      end = tmp;
      state.advance();
      finished = true;
      break;
    }
    if (!tripleQuote)
      literal.push_back(*state);
    state.advance();
  }
  if (!finished) {
    emitError(state.getLoc(), "unterminated string literal");
    return Invalid;
  }
  spelling = llvm::StringRef(begin, end - begin);
  if (!tripleQuote && escaped) {
    stringLiterals.push_back(literal.str().str());
    spelling = stringLiterals.back();
  }
  return StringLiteral;
}
} // namespace xblang
