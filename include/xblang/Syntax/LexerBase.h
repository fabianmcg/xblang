//===- LexerBase.cpp - XBLang lexer classes  ---------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the base XBLang lexer classes.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_SYNTAX_LEXERBASE_H
#define XBLANG_SYNTAX_LEXERBASE_H

#include "mlir/IR/Diagnostics.h"
#include "xblang/Basic/SourceManager.h"
#include "xblang/Basic/SourceState.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_os_ostream.h"

namespace xblang {
class SyntaxContext;

namespace syntax {
class LexerBase;

//===----------------------------------------------------------------------===//
// Token
//===----------------------------------------------------------------------===//
/// Base class for representing tokens.
class Token {
public:
  /// Constant indicating the end of file.
  static constexpr int EndOfFile = 0;
  /// Constant indicating the token is not valid.
  static constexpr int Invalid = -1;
  /// Constant indicating the token is not initialized.
  static constexpr int Uninitialized = -2;
  // Constructors
  Token() = default;

  Token(int token, SourceString spelling) : token(token), spelling(spelling) {}

  Token(int token, llvm::StringRef spelling, SourceLocation loc)
      : token(token),
        spelling(SourceString(
            spelling.empty() ? llvm::StringRef(loc.getLoc(), 0) : spelling,
            loc.line, loc.column)) {}

  /// Returns true if the token is uninitialized.
  inline bool isUninitialized() const { return token == Uninitialized; }

  /// Returns true if the token is invalid.
  inline bool isInvalid() const { return token == Invalid; }

  /// Returns true if the token is the end of file.
  inline bool isEndOfFile() const { return token == EndOfFile; }

  /// Returns true if the token is not uninitialized or invalid.
  inline bool isValid() const { return !isUninitialized() && !isInvalid(); }

  /// Returns true if the tokens are the same.
  inline bool operator==(int tok) const { return tok == token; }

  /// Returns true if the tokens are the same.
  inline bool operator==(const Token &tok) const {
    return tok == token && spelling == tok.spelling;
  }

  /// Returns the token value.
  inline int getTok() const { return token; }

  /// Returns the spelling of the token.
  inline llvm::StringRef getSpelling() const { return spelling.str(); }

  /// Returns the location of the token.
  inline SourceLocation getLoc() const { return spelling.getLoc(); }

  /// Sets the token value to `tok.`
  void setTok(int tok) { token = tok; }

protected:
  /// Token identifier value.
  int token = Uninitialized;
  /// Token spelling.
  SourceString spelling;
};

//===----------------------------------------------------------------------===//
// LexerToken
//===----------------------------------------------------------------------===//
/// Class for representing tokens from a specific lexer.
template <typename Lexer>
class LexerToken : public Token {
public:
  using Token::Token;

  LexerToken(Token token) : Token(token) {}

  /// Compares tokens from the same lexer.
  inline bool operator==(const LexerToken &tok) const {
    return Token::operator==(tok);
  }

  /// Returns the token value using the enum type TokenID.
  inline auto getTok() const { return Lexer::getToken(token); }

  /// Implicitly convert to an int.
  constexpr inline operator int() const { return token; }
};

//===----------------------------------------------------------------------===//
// LexerGuard
//===----------------------------------------------------------------------===//
/// RAII guard that sets the internal state of the lexer on destruction.
class ParserGuard;

class LexerGuard {
public:
  LexerGuard(LexerBase &lexer);
  ~LexerGuard();

  /// Releases the guard.
  void release();

protected:
  friend class ParserGuard;
  /// Resets the state of the lexer.
  void reset();
  LexerBase *lexer{};
  SourceState state{};
  Token token;
};

//===----------------------------------------------------------------------===//
// LexerBase
//===----------------------------------------------------------------------===//
/// Base class for all lexers.
class LexerBase : public SMDiagnosticsEmitter {
public:
  friend class xblang::SyntaxContext;
  friend class xblang::syntax::LexerGuard;

  LexerBase(SourceManager &srcManager) : SMDiagnosticsEmitter(&srcManager) {}

  virtual ~LexerBase() = default;

  /// Resets the source state to the provided state, and consumes a token.
  LexerBase &operator()(const SourceState &state, bool consume = true);
  LexerBase &operator()(const SourceLocation &loc, bool consume = true);
  LexerBase &operator()(const std::pair<SourceLocation, const char *> &state,
                        bool consume = true);

  /// Returns true if the lexer is in a valid state.
  bool isValid() const { return state.isValid(); }

  /// Returns true if the lexer is at the end of the file.
  bool isEndOfFile() const {
    return token.isEndOfFile() || state.isEndOfFile();
  }

  /// Returns the location of the current token.
  SourceLocation getLoc() const { return token.getLoc(); }

  /// Returns the pointer to the buffer.
  const char *getBuf() const { return state.begin(); }

  /// Returns a checkpoint to the current state.
  SourceLocation checkpoint() const { return getLoc(); }

  /// Returns apointer to the end of the source buffer.
  const char *getEnd() const { return state.end(); }

  /// Returns a guard to the current state.
  LexerGuard getGuard() { return LexerGuard(*this); }

  LexerGuard getGuard(const SourceState &state) {
    auto guard = LexerGuard(*this);
    (*this)(state);
    return guard;
  }

  /// Returns the spelling
  static llvm::StringRef getSpelling(const SourceLocation &start,
                                     const SourceLocation &end);
  static llvm::StringRef getSpelling(const SourceLocation &start,
                                     const SourceLocation &end, size_t startOff,
                                     size_t endOff);

  /// Returns the source state.
  SourceState &getState() { return state; }

protected:
  /// Consume a token.
  virtual Token consumeToken() = 0;
  /// Source state.
  SourceState state{};
  /// Current token.
  Token token;
};

//===----------------------------------------------------------------------===//
// LexerMixin
//===----------------------------------------------------------------------===//
/// Utility class for defining lexers using the CRTP idiom.
template <typename Derived>
class LexerMixin : public LexerBase {
public:
  using Base = LexerMixin<Derived>;
  using Lexer = Derived;
  using LexerToken = xblang::syntax::LexerToken<Lexer>;
  friend class xblang::SyntaxContext;

  LexerMixin(SourceManager &srcManager) : LexerBase(srcManager) {
    getDerived().registerKeywords();
  }

  /// Returns the lexer id.
  static int getID() { return id; }

  /// Returns the current token.
  LexerToken getTok() const { return token; }

  /// Consumes a token and returns it.
  LexerToken consume() { return consumeToken(); }

  /// Lexes a token from the source.
  LexerToken lex(SourceState &state) const {
    auto token = getDerived().lexMain(state);
    if (token.getTok() == Lexer::Identifier)
      token.setTok(getKeyword(token.getSpelling()));
    if (token.isInvalid())
      emitError(token.getLoc(), "lexing error: invalid token found.");
    return token;
  }

  /// Returns the TokenID for the keyword `key`.
  auto getKeyword(llvm::StringRef key) const {
    auto it = keywords.find(key);
    if (it != keywords.end())
      return Lexer::getToken(it->second);
    return Lexer::getToken(Lexer::Identifier);
  }

protected:
  Lexer &getDerived() { return static_cast<Lexer &>(*this); }

  const Lexer &getDerived() const { return static_cast<const Lexer &>(*this); }

  /// Consumes a token from the source.
  Token consumeToken() override {
    if (state.isValid())
      token = getDerived().lex(state);
    else
      return LexerToken();
    return token;
  }

  void addKeyword(int tok, llvm::StringRef id) {
    (void)keywords.insert({id, tok});
  }

  void registerKeywords() {}

private:
  /// Lexer ID.
  static inline int id = -1;
  /// Keywords used in the lexer.
  llvm::StringMap<int> keywords;
};
} // namespace syntax
} // namespace xblang
#endif
