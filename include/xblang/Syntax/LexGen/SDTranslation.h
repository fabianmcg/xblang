//===- SDTranslation.h - Lex syntax-directed translator ----------*- C++-*-===//
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

#ifndef XBLANG_SYNTAX_LEXGEN_SDTRANSLATION_H
#define XBLANG_SYNTAX_LEXGEN_SDTRANSLATION_H

#include "mlir/IR/Value.h"
#include "xblang/Basic/SourceState.h"
#include "xblang/Syntax/LexerBase.h"
#include "xblang/Syntax/ParseResult.h"
#include "xblang/Syntax/Utils/SDTUtils.h"
#include "llvm/Support/ErrorOr.h"

namespace mlir {
class OpBuilder;
}

namespace xblang {
class Source;
} // namespace xblang

namespace xblang {
namespace syntaxgen {
namespace lex {
/// Syntax-directed lexer for the lexer generator grammar.
class SDTLexer : public syntax::LexerMixin<SDTLexer>,
                 public syntaxgen::SDTLexerCommon {
public:
  template <typename>
  friend class syntax::LexerMixin;
  using base = syntax::LexerMixin<SDTLexer>;
  using Token = typename base::LexerToken;
  using base::LexerMixin;

  /// Tokens.
  typedef enum {
    Invalid = Token::Invalid,
    EndOfFile = Token::EndOfFile,

#define CHAR_PUNCTUATION(_0, _1) _0 = _1,
#include "Tokens.inc"
    LastChar = 256,

#define CHAR_PUNCTUATION(...)
#define TERMINAL(_0, ...) _0,
#include "Tokens.inc"
  } TokenID;

  /// Returns an int as a token ID.
  static inline constexpr TokenID getToken(int value) {
    return static_cast<TokenID>(value);
  }

  /// Converts a token ID to a string representation.
  static llvm::StringRef toString(TokenID value);

  /// Main lexing routine.
  TokenID lexMain(SourceState &state, SourceLocation &beginLoc,
                  llvm::StringRef &spelling) const;

  inline Token lexMain(SourceState &state) const {
    SourceLocation loc;
    llvm::StringRef spelling;
    auto tok = lexMain(state, loc, spelling);
    return Token(tok, spelling, loc);
  }

  /// Returns the last character lexed.
  uint32_t getCharacter() const { return character; }

  /// Changes the lexing mode to lex single characters.
  void setLexChars(bool value) { lexChars = value; }

private:
  /// Lex a single character.
  TokenID lexChar(SourceState &state, llvm::StringRef &spelling,
                  uint32_t &character) const;

  /// Lex a punctuation terminal.
  TokenID lexPunctuation(SourceState &state, llvm::StringRef &spelling) const;

  /// Registers the lexer keywords.
  void registerKeywords() { addKeyword(EmptyString, "eps"); }

  /// Character lexed during `lexChar`.
  mutable uint32_t character{};
  /// State indicating whether to lex the inputs using `lexChar`.
  bool lexChars = false;
};

/// Class for performing syntax-directed translation to Lex IR.
class SDTranslator {
private:
  using ParseOp = syntax::ParseResult<mlir::Operation *>;
  using ParseValue = syntax::ParseResult<mlir::Value>;
  using Lexer = SDTLexer;

public:
  SDTranslator(Source &source);
  ~SDTranslator();
  /// Parse a lexing rule and add it to the end of the block.
  syntax::ParseResult<mlir::Operation *> parseRule(llvm::StringRef id,
                                                   mlir::Block &block);
  /// Parse a lexing definition and add it to the end of the block.
  syntax::ParseResult<mlir::Operation *> parseDefinition(llvm::StringRef id,
                                                         mlir::Block &block);

protected:
  /// Parse a lexing expression and add it to the end of the block.
  syntax::ParseResult<mlir::Value> parseExpr(mlir::Value lhs,
                                             mlir::OpBuilder &builder);
  /// Parse a lexing top expression and add it to the end of the block.
  syntax::ParseResult<mlir::Value> parseTopExpr(mlir::OpBuilder &builder);
  /// Parse a lexing parenthesis expression and add it to the end of the block.
  syntax::ParseResult<mlir::Value> parseParenExpr(mlir::OpBuilder &builder);
  /// Parse a lexing literal expression and add it to the end of the block.
  syntax::ParseResult<mlir::Value> parseLiteralExpr(mlir::OpBuilder &builder);
  /// Parse a lexing character class expression and add it to the end of the
  /// block.
  syntax::ParseResult<mlir::Value> parseCharClassExpr(mlir::OpBuilder &builder);
  /// Parse a lexing symbol reference expression and add it to the end of the
  /// block.
  syntax::ParseResult<mlir::Value> parseDefRefExpr(mlir::OpBuilder &builder);

  /// Matches an input token against an expected token.
  bool match(Lexer::TokenID tok, Lexer::TokenID expected);
  std::pair<bool, SourceLocation> matchAndConsume(Lexer::TokenID expected);

private:
  /// Input source source.
  Source &source;
  /// Source state.
  SourceState state;
  /// Internal lexer.
  Lexer lexer;
};
} // namespace lex
} // namespace syntaxgen
} // namespace xblang

#endif // XBLANG_SYNTAX_LEXGEN_SDTRANSLATION_H
