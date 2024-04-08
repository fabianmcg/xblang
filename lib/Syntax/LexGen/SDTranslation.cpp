//===- SDTranslation.cpp - Lex syntax-directed translator  -------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the syntax-directed translator for the Lex IR.
//
//===----------------------------------------------------------------------===//

#include "xblang/Syntax/LexGen/SDTranslation.h"
#include "mlir/IR/Builders.h"
#include "xblang/Syntax/IR/SyntaxDialect.h"
#include "xblang/Syntax/LexerBase.h"

using namespace xblang;
using namespace xblang::syntax;
using namespace xblang::syntaxgen::lex;

//===----------------------------------------------------------------------===//
// SDTLexer
//===----------------------------------------------------------------------===//
llvm::StringRef SDTLexer::toString(TokenID value) {
  switch (value) {
#define TERMINAL(_0, ...)                                                      \
  case TokenID::_0:                                                            \
    return #_0;
#include "xblang/Syntax/LexGen/Tokens.inc"
  default:
    return {};
  }
}

SDTLexer::TokenID SDTLexer::lexMain(SourceState &state,
                                    SourceLocation &beginLoc,
                                    llvm::StringRef &spelling) const {
  if (state.isEndOfFile())
    return TokenID::EndOfFile;
  auto fallBackState = state;
  TokenID tok = TokenID::Invalid;
  while (true) {
    auto character = *state;
    if (!character)
      return TokenID::Invalid;
    if (lexChars)
      return lexChar(state, spelling, this->character);
    consumeSpaces(state, character);
    fallBackState = state;
    beginLoc = state.getLoc();
    if (!character) {
      tok = TokenID::EndOfFile;
      break;
    }
    // Lex comment
    if (character == '/' && state.at(1) == '*') {
      std::optional<llvm::StringRef> comment = consumeComment(state);
      if (!comment) {
        emitError(beginLoc, "the comment was never closed");
        tok = TokenID::Invalid;
        break;
      }
      tok = TokenID::Comment;
      continue;
    }
    // Lex keyword or identifier
    if (isalpha(character) || character == '_') {
      spelling = consumeIdentifier(state);
      tok = TokenID::Identifier;
      break;
    } /* Lex number */ else if (isdigit(character)) {
      spelling = consumeNumber(state);
      tok = TokenID::Int;
      break;
    } /* Lex punctuation */ else if (ispunct(character)) {
      tok = lexPunctuation(state, spelling);
      break;
    }
    tok = TokenID::Invalid;
    break;
  }
  return tok;
}

SDTLexer::TokenID SDTLexer::lexChar(SourceState &state,
                                    llvm::StringRef &spelling,
                                    uint32_t &character) const {
  auto buffer = state.begin();
  std::string error;
  SourceLocation loc;
  struct Char c = consumeChar(state, loc, error);
  spelling = llvm::StringRef(buffer, state.begin() - buffer);
  if (c.state == Char::Invalid)
    emitError(loc, error);
  character = c.character;
  if (c.isUTF())
    return TokenID::Char;
  if (character == ']')
    return TokenID::RBracket;
  else if (character == '-')
    return TokenID::Dash;
  else if (character == '\'')
    return TokenID::Quote;
  return TokenID::Char;
}

SDTLexer::TokenID SDTLexer::lexPunctuation(SourceState &state,
                                           llvm::StringRef &spelling) const {
  SourceState tmp = state;
  while (ispunct(tmp.advance().get()))
    ;
  size_t len = static_cast<size_t>(tmp.begin() - state.begin());
  TokenID tok = TokenID::Invalid;
  spelling = llvm::StringRef(state.begin(), len);
  if (len > 1)
    tok = llvm::StringSwitch<TokenID>(spelling)
#define CHAR_PUNCTUATION(_0, _1)
#define PUNCTUATION(_0, _1) .Case(_1, TokenID::_0)
#include "xblang/Syntax/LexGen/Tokens.inc"
              .Default(TokenID::Invalid);
  if (tok == TokenID::Invalid) {
    switch (state.get()) {
#define CHAR_PUNCTUATION(_0, _1)                                               \
  case _1:                                                                     \
    tok = TokenID::_0;                                                         \
    break;
#include "xblang/Syntax/LexGen/Tokens.inc"
    default:
      tok = TokenID::Invalid;
    }
    spelling = llvm::StringRef(state.begin(), 1);
    ++state;
  } else if (len > 0)
    while (len-- > 0)
      state.advance();
  return tok;
}

//===----------------------------------------------------------------------===//
// SDTranslator
//===----------------------------------------------------------------------===//
SDTranslator::SDTranslator(Source &source)
    : source(source), state(source.getState()),
      lexer(*source.getSourceManager()) {
  if (state.isValid())
    lexer(state);
}

SDTranslator::~SDTranslator() = default;

bool SDTranslator::match(Lexer::TokenID tok, Lexer::TokenID expected) {
  if (tok != expected) {
    source.emitError(lexer.getLoc(), "expected: '" + lexer.toString(expected) +
                                         "', but got: '" + lexer.toString(tok) +
                                         "'");
    return false;
  }
  return true;
}

std::pair<bool, SourceLocation>
SDTranslator::matchAndConsume(Lexer::TokenID expected) {
  auto tok = lexer.getTok();
  auto loc = lexer.getLoc();
  auto result = match(tok.getTok(), expected);
  if (result)
    lexer.consume();
  return {result, loc};
}

ParseResult<mlir::Operation *> SDTranslator::parseRule(llvm::StringRef id,
                                                       mlir::Block &block) {
  mlir::OpBuilder builder(&block, block.end());
  auto def = builder.create<RuleOp>(source.getLoc(lexer.getLoc()), id);
  def.getBodyRegion().emplaceBlock();
  builder.setInsertionPoint(def.getBody(0), def.getBody(0)->end());
  auto regex = parseExpr(nullptr, builder);
  if (!regex.isSuccess())
    return ParseOp::error(regex.getLoc());
  builder.create<ReturnOp>(source.getLoc(regex.getLoc()), regex.get());
  return ParseOp::success(def, regex.getLoc());
}

syntax::ParseResult<mlir::Operation *>
SDTranslator::parseDefinition(llvm::StringRef id, mlir::Block &block) {
  mlir::OpBuilder builder(&block, block.end());
  auto def = builder.create<MacroOp>(source.getLoc(lexer.getLoc()), id);
  builder.setInsertionPoint(def.getBody(0), def.getBody(0)->end());
  auto regex = parseExpr(nullptr, builder);
  if (!regex.isSuccess())
    return ParseOp::error(regex.getLoc());
  builder.create<ReturnOp>(source.getLoc(regex.getLoc()), regex.get());
  return ParseOp::success(def, regex.getLoc());
}

ParseResult<mlir::Value> SDTranslator::parseExpr(mlir::Value lhs,
                                                 mlir::OpBuilder &builder) {
  mlir::Value expr = nullptr;
  auto bLoc = lexer.getLoc();
  while (true) {
    auto tok = lexer.getTok();
    if (tok == Lexer::Invalid)
      return ParseValue::error(lexer.getLoc());
    if (tok == Lexer::EndOfFile || tok == Lexer::BOr)
      break;
    auto subExpr = parseTopExpr(builder);
    if (subExpr.isEmpty())
      break;
    if (!subExpr.isSuccess())
      return subExpr;
    if (!expr)
      expr = subExpr.get();
    else
      expr = builder.create<AndOp>(expr.getLoc(), expr, subExpr.get());
  }
  if (!expr) {
    lexer.emitError(bLoc, "empty expressions are not valid");
    return ParseValue::error(bLoc);
  }
  if (lhs)
    expr = builder.create<OrOp>(expr.getLoc(), lhs, expr);
  if (lexer.getTok() == Lexer::BOr) {
    lexer.consume();
    return parseExpr(expr, builder);
  }
  return ParseValue::success(std::move(expr), bLoc);
}

syntax::ParseResult<mlir::Value>
SDTranslator::parseTopExpr(mlir::OpBuilder &builder) {
  ParseValue expr;
  auto bLoc = lexer.getLoc();
  switch (lexer.getTok()) {
  case Lexer::LParen:
    expr = parseParenExpr(builder);
    if (!expr.isSuccess())
      return expr;
    break;
  case Lexer::LBracket:
    expr = parseCharClassExpr(builder);
    if (!expr.isSuccess())
      return expr;
    break;
  case Lexer::Quote:
    expr = parseLiteralExpr(builder);
    if (!expr.isSuccess())
      return expr;
    break;
  case Lexer::Identifier:
    expr = parseDefRefExpr(builder);
    if (!expr.isSuccess())
      return expr;
    break;
  case Lexer::EmptyString: {
    lexer.consume();
    expr.get() = builder.create<EmptyStringOp>(source.getLoc(bLoc));
    break;
  }
  default:
    break;
  }
  while (true) {
    switch (lexer.getTok()) {
    case Lexer::Multiply: {
      lexer.consume();
      expr.get() = builder.create<ZeroOrMoreOp>(source.getLoc(bLoc), expr.get(),
                                                nullptr);
      break;
    }
    case Lexer::Plus: {
      lexer.consume();
      auto zeroOrMore =
          builder.create<ZeroOrMoreOp>(source.getLoc(bLoc), expr.get());
      expr.get() =
          builder.create<AndOp>(source.getLoc(bLoc), expr.get(), zeroOrMore);
      break;
    }
    case Lexer::Question: {
      lexer.consume();
      expr.get() = builder.create<OrOp>(
          source.getLoc(bLoc), expr.get(),
          builder.create<EmptyStringOp>(builder.getUnknownLoc()));
      break;
    }
    default:
      return expr;
    }
  }
  return expr;
}

syntax::ParseResult<mlir::Value>
SDTranslator::parseParenExpr(mlir::OpBuilder &builder) {
  auto lparen = matchAndConsume(Lexer::LParen);
  if (!lparen.first)
    return ParseValue::error(lparen.second);
  auto expr = parseExpr(nullptr, builder);
  if (!expr.isSuccess()) {
    source.emitError(expr.getLoc(), "expected a non empty expression");
    return ParseValue::error(expr.getLoc());
  }
  auto rparen = matchAndConsume(Lexer::RParen);
  if (!rparen.first)
    return ParseValue::error(rparen.second);
  return ParseValue::success(std::move(expr.get()), lparen.second);
}

syntax::ParseResult<mlir::Value>
SDTranslator::parseLiteralExpr(mlir::OpBuilder &builder) {
  auto bLoc = lexer.getLoc();
  auto matchResult = match(lexer.getTok().getTok(), Lexer::Quote);
  if (!matchResult)
    return ParseValue::error(bLoc);
  lexer.setLexChars(true);
  lexer.consume();
  mlir::Value expr{};
  while (true) {
    auto tok = lexer.getTok();
    if (tok == Lexer::Quote) {
      lexer.setLexChars(false);
      lexer.consume();
      break;
    }
    if (tok == Lexer::Invalid)
      return ParseValue::error(lexer.getLoc());
    if (tok == Lexer::EndOfFile) {
      source.emitError(
          lexer.getLoc(),
          "unfinished literal, every literal has to be closed with `'`");
      return ParseValue::error(lexer.getLoc());
    }
    auto loc = source.getLoc(lexer.getLoc());
    uint32_t character = lexer.getCharacter();
    lexer.consume();
    if (!expr)
      expr = builder.create<TerminalOp>(
          loc, builder.getAttr<LiteralAttr>(character));
    else
      expr = builder.create<AndOp>(
          loc, expr,
          builder.create<TerminalOp>(loc,
                                     builder.getAttr<LiteralAttr>(character)));
  }
  if (!expr) {
    source.emitError(lexer.getLoc(), "literals cannot be null");
    return ParseValue::error(lexer.getLoc());
  }
  return ParseValue::success(std::move(expr), bLoc);
}

syntax::ParseResult<mlir::Value>
SDTranslator::parseCharClassExpr(mlir::OpBuilder &builder) {
  auto bLoc = lexer.getLoc();
  auto matchResult = match(lexer.getTok().getTok(), Lexer::LBracket);
  if (!matchResult)
    return ParseValue::error(bLoc);
  lexer.setLexChars(true);
  lexer.consume();
  CharClass charClass;
  auto isChar = [](Token tok) {
    return tok == Lexer::Char || tok == Lexer::Quote;
  };
  while (lexer.isValid() && !lexer.getTok().isEndOfFile()) {
    auto tok = lexer.getTok();
    if (tok == Lexer::RBracket)
      break;
    uint32_t l = lexer.getCharacter();
    if (isChar(tok)) {
      lexer.consume();
      if (lexer.getTok() == Lexer::Dash) {
        lexer.consume();
        uint32_t u = lexer.getCharacter();
        matchAndConsume(Lexer::Char);
        charClass.insert(l, u);
        continue;
      }
      charClass.insert(l);
      continue;
    }
    return ParseValue::error(bLoc);
  }
  lexer.setLexChars(false);
  auto rBracket = matchAndConsume(Lexer::RBracket);
  if (!rBracket.first)
    return ParseValue::error(rBracket.second);
  charClass.computeUSR();
  return ParseValue::success(
      builder.create<TerminalOp>(
          source.getLoc(bLoc),
          builder.getAttr<CharClassAttr>(std::move(charClass))),
      bLoc);
}

syntax::ParseResult<mlir::Value>
SDTranslator::parseDefRefExpr(mlir::OpBuilder &builder) {
  auto tok = lexer.getTok();
  auto spelling = tok.getSpelling();
  auto result = matchAndConsume(Lexer::Identifier);
  if (!result.first)
    return ParseValue::error(result.second);
  return ParseValue::success(
      builder.create<CallOp>(source.getLoc(tok.getLoc()), spelling),
      tok.getLoc());
}
