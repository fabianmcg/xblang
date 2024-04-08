//===- SDTranslation.h - Parse syntax-directed translator --------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the syntax-directed translator for the Parse IR.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_SYNTAX_PARSERGEN_SDTRANSLATION_H
#define XBLANG_SYNTAX_PARSERGEN_SDTRANSLATION_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "xblang/Basic/SourceState.h"
#include "xblang/Syntax/LexerBase.h"
#include "xblang/Syntax/ParseResult.h"
#include "xblang/Syntax/Utils/SDTUtils.h"
#include "xblang/TableGen/Lexer.h"
#include "llvm/Support/ErrorOr.h"

namespace mlir {
class OpBuilder;
}

namespace xblang {
class Source;
} // namespace xblang

namespace xblang {
namespace syntaxgen {
namespace parser {
/// Terminal map used by the parser.
class TerminalMap {
public:
  using RecordTy = const llvm::Record *;
  TerminalMap(tablegen::Lexer lexer, llvm::StringRef unspecifiedKey = "");

  /// Returns the lexer construct named by key.
  RecordTy get(llvm::StringRef key) const {
    auto it = tokenMap.find(key);
    if (it != tokenMap.end())
      return it->second;
    return nullptr;
  }

  /// Returns the default construct to use if a key is not in the map.
  RecordTy getDefaultConstruct() const { return defaultConstruct; }

  /// Returns whether the def is not in the token map.
  static bool isUnspecified(RecordTy def) { return def == nullptr; }

private:
  llvm::StringMap<RecordTy> tokenMap{};
  tablegen::Lexer lexer;
  llvm::StringRef unspecifiedKey;
  RecordTy defaultConstruct{};
};

/// Syntax-directed lexer for the parser generator grammar.
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

  /// Main parsing routine.
  TokenID lexMain(SourceState &state, SourceLocation &beginLoc,
                  llvm::StringRef &spelling) const;

  inline Token lexMain(SourceState &state) const {
    SourceLocation loc;
    llvm::StringRef spelling;
    auto tok = lexMain(state, loc, spelling);
    return Token(tok, spelling, loc);
  }

private:
  /// Lex a punctuation terminal.
  TokenID lexPunctuation(SourceState &state, llvm::StringRef &spelling) const;

  /// Registers the lexer keywords.
  void registerKeywords() { addKeyword(EmptyString, "eps"); }
};

/// Class for performing syntax-directed translation to Parse IR.
class SDTranslator {
private:
  using ParseOp = syntax::ParseResult<mlir::Operation *>;
  using ParseValue = syntax::ParseResult<mlir::Value>;
  using ParseAttr = syntax::ParseResult<mlir::Attribute>;
  using Lexer = SDTLexer;

public:
  SDTranslator(SourceManager &sourceManager, TerminalMap &tokMap,
               mlir::SymbolTable &symTable);
  ~SDTranslator();
  /// Parse a parsing rule and add it to the end of the block.
  syntax::ParseResult<mlir::Operation *>
  parseProduction(Source *source, llvm::StringRef id, mlir::Block &block);

  /// Parse a macro and add it to the end of the block.
  syntax::ParseResult<mlir::Operation *>
  parseMacro(Source *source, const std::vector<llvm::StringRef> &args,
             llvm::StringRef id, mlir::Block &block);

protected:
  /// Parse a parsing expression and add it to the end of the block.
  syntax::ParseResult<mlir::Value> parseExpr(mlir::Value lhs, bool parseAsRule,
                                             mlir::OpBuilder &builder,
                                             size_t &sid);
  /// Parse a parsing top expression and add it to the end of the block.
  syntax::ParseResult<mlir::Value> parseTopExpr(mlir::OpBuilder &builder,
                                                size_t &sid);
  /// Parse a parsing parenthesis expression and add it to the end of the block.
  syntax::ParseResult<mlir::Value> parseParenExpr(mlir::OpBuilder &builder,
                                                  size_t &sid);
  /// Parse a parsing literal expression and add it to the end of the block.
  syntax::ParseResult<mlir::Value> parseTerminalExpr(mlir::OpBuilder &builder,
                                                     size_t &sid);
  /// Parse a non-terminal expression and add it to the end of the block.
  syntax::ParseResult<mlir::Value>
  parseNonTerminalExpr(mlir::OpBuilder &builder, size_t &sid);
  /// Parse a call expression and add it to the end of the block.
  syntax::ParseResult<mlir::Value> parseCallExpr(mlir::OpBuilder &builder,
                                                 size_t &sid);
  /// Parse a syntax directive and adds it to the end of the block.
  syntax::ParseResult<mlir::Value> parseDirective(mlir::OpBuilder &builder,
                                                  size_t &sid);
  /// Parse a name attribute.
  syntax::ParseResult<mlir::FlatSymbolRefAttr>
  parseNameAttr(mlir::OpBuilder &builder);
  /// Matches an input token against an expected token.
  bool match(Lexer::TokenID tok, Lexer::TokenID expected);
  std::pair<bool, SourceLocation> matchAndConsume(Lexer::TokenID expected);

private:
  /// Source manager.
  SourceManager &sourceManager;
  /// Token map.
  TerminalMap &tokMap;
  /// Source state.
  SourceState state;
  /// Internal lexer.
  Lexer lexer;
  /// Argument mapper.
  llvm::StringMap<mlir::Value> arguments;
  /// Symbol table for the parser module.
  mlir::SymbolTable &symTable;
  /// Symbol counter.
  size_t symId = 0;
};
} // namespace parser
} // namespace syntaxgen
} // namespace xblang

#endif // XBLANG_SYNTAX_PARSERGEN_SDTRANSLATION_H
