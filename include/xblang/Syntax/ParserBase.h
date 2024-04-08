//===- ParserBase.cpp - Parser base classes ----------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the base classes for defining all parsers.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_SYNTAX_PARSERBASE_H
#define XBLANG_SYNTAX_PARSERBASE_H

#include "xblang/Basic/TypeInfo.h"
#include "xblang/Syntax/LexerBase.h"
#include "xblang/Syntax/ParseResult.h"
#include "llvm/ADT/DenseMap.h"
#include <functional>
#include <map>
#include <memory>

namespace xblang {
class SyntaxContext;

namespace syntax {
class ParserBase;
class DynamicParser;

//===----------------------------------------------------------------------===//
// DynParsingCombinator
//===----------------------------------------------------------------------===//
/// Struct for representing dynamic parsing combinators.
struct DynParsingCombinator {
  /// Type for representing dynamic parsing combinators.
  using ParsingFn = llvm::function_ref<void(ParserBase *, SourceState &,
                                            ParsingStatus &, bool)>;
  /// Parser instance owning the combinator.
  ParserBase *parser{};
  /// THe free parsing combinator.
  ParsingFn combinator{};

  operator bool() const { return parser && combinator; }
};

//===----------------------------------------------------------------------===//
// DynParsingScope
//===----------------------------------------------------------------------===//
/// Dynamic parsing scope.
class DynParsingScope {
public:
  using KeywordID = llvm::StringRef;
  using Key = std::pair<KeywordID, TypeInfo::ID>;
  using Value = DynParsingCombinator;
  using Map = llvm::DenseMap<Key, Value>;

private:
  DynamicParser *dynParser{};
  DynParsingScope *prevScope{};
  llvm::SmallVector<std::pair<Key, Value>, 4> guard{};

public:
  DynParsingScope() = delete;
  DynParsingScope(DynParsingScope &&) = default;
  DynParsingScope(const DynParsingScope &) = delete;
  DynParsingScope(DynamicParser &parser);
  ~DynParsingScope();

  /// Inserts a new value into the current scope.
  void insert(Key key, Value val);
};

//===----------------------------------------------------------------------===//
// DynamicParser
//===----------------------------------------------------------------------===//
/// Class implementing a dynamic parser.
class DynamicParser {
private:
  friend class DynParsingScope;
  using KeywordID = DynParsingScope::KeywordID;
  using ParsingFn = DynParsingCombinator::ParsingFn;
  using Map = DynParsingScope::Map;
  Map parserTable{};
  DynParsingScope rootScope;
  DynParsingScope *currentScope{};

public:
  DynamicParser() : rootScope(*this) { currentScope = &rootScope; }

  /// Checks if a token is a dynamic combinator.
  bool isTok(const Token &tok, TypeInfo combinatorID) const {
    return parserTable.count({tok.getSpelling(), combinatorID.getID()});
  }

  /// Returns a parser combinator given a protected keyword.
  DynParsingCombinator getCombinator(llvm::StringRef key,
                                     TypeInfo combinatorID) const {
    auto it = parserTable.find({key, combinatorID.getID()});
    if (it == parserTable.end())
      return {};
    return it->second;
  }

  /// Pushes down a context scope.
  DynParsingScope pushScope() { return DynParsingScope(*this); }

  /// Registers a combinator within the current scope.
  void registerCombinator(llvm::StringRef keyword, TypeInfo combinatorID,
                          ParserBase *parser, ParsingFn combinator) {
    assert(currentScope && "invalid active scope");
    currentScope->insert({keyword, combinatorID.getID()}, {parser, combinator});
  }
};

//===----------------------------------------------------------------------===//
// ParserGuard
//===----------------------------------------------------------------------===//
class ParserGuard : public LexerGuard, public PureParsingStatus {
public:
  using PureParsingStatus::operator bool;
  using PureParsingStatus::operator=;

  ParserGuard(LexerBase &lexer, bool emitErrors)
      : LexerGuard(lexer), PureParsingStatus(Empty), emitErrors(emitErrors) {}

  ~ParserGuard() {
    if (isSuccess())
      release();
  }

  /// Resets the state.
  void reset() {
    status = Empty;
    LexerGuard::reset();
  }

  /// Updates the guard.
  void update(ParserGuard &&grd) {
    token = std::exchange(grd.token, Token());
    state = std::exchange(grd.state, SourceState());
    lexer = std::exchange(grd.lexer, nullptr);
    status = std::exchange(grd.status, Empty);
  }

  /// Returns the current location.
  SourceLocation getLoc() const { return state.getLoc(); }

  /// Returns whether errors should be emitted.
  bool getEmitErrors() const { return emitErrors; }

private:
  bool emitErrors;
};

//===----------------------------------------------------------------------===//
// ParserBase
//===----------------------------------------------------------------------===//
/// Base class for all parsers.
class ParserBase : public SMDiagnosticsEmitter {
public:
  friend class xblang::SyntaxContext;
  using SMDiagnosticsEmitter::getLoc;

  ParserBase(SourceManager &srcManager) : SMDiagnosticsEmitter(&srcManager) {}

  /// Creates a new empty parsing result.
  template <typename T>
  static ParseResult<T> empty(const SourceLocation &loc = {}) {
    return ParseResult<T>::empty(loc);
  }

  static ParsingStatus empty(const SourceLocation &loc) {
    return ParsingStatus::empty(loc);
  }

  static PureParsingStatus empty() { return PureParsingStatus::Empty; }

  /// Creates a new success parsing result.
  template <typename T>
  static ParseResult<T> success(T &&value, const SourceLocation &loc) {
    return ParseResult<T>::success(std::forward<T>(value), loc);
  }

  static ParsingStatus success(const SourceLocation &loc) {
    return ParsingStatus::success(loc);
  }

  static PureParsingStatus success() { return PureParsingStatus::Success; }

  /// Creates a new error parsing result.
  template <typename T>
  static ParseResult<T> error(const SourceLocation &loc = {}) {
    return ParseResult<T>::error(loc);
  }

  static ParsingStatus error(const SourceLocation &loc) {
    return ParsingStatus::error(loc);
  }

  static PureParsingStatus error() { return PureParsingStatus::Error; }

  /// Creates a new fatal parsing result.
  template <typename T>
  static ParseResult<T> fatal(const SourceLocation &loc = {}) {
    return ParseResult<T>::fatal(loc);
  }

  static ParsingStatus fatal(const SourceLocation &loc) {
    return ParsingStatus::fatal(loc);
  }

  static PureParsingStatus fatal() { return PureParsingStatus::Fatal; }

  /// Returns a location suitable to be used in MLIR.
  mlir::Location getLoc(const ParsingStatus &loc,
                        bool useFileLineColFallback = false) const {
    return SMDiagnosticsEmitter::getLoc(loc.getLoc(), useFileLineColFallback);
  }

  mlir::Location getLoc(const Token &tok,
                        bool useFileLineColFallback = false) const {
    return SMDiagnosticsEmitter::getLoc(tok.getLoc(), useFileLineColFallback);
  }

  /// Determines whether a token is a token for a dynamic combinator.
  bool isDynTok(const Token &tok, TypeInfo info) const {
    if (!dynParser)
      return false;
    return dynParser->isTok(tok, info);
  }

  /// Returns a parser combinator given a protected keyword.
  DynParsingCombinator getCombinator(llvm::StringRef key, TypeInfo info) const {
    if (!dynParser)
      return {};
    return dynParser->getCombinator(key, info);
  }

  DynParsingCombinator getCombinator(const Token &tok, TypeInfo info) const {
    return getCombinator(tok.getSpelling(), info);
  }

  /// Pushes down a context scope.
  DynParsingScope pushScope() {
    assert(dynParser && "null dynamic parser");
    return dynParser->pushScope();
  }

  /// Sets the dynamic parser.
  void setDynamicParser(DynamicParser *dynParser) {
    this->dynParser = dynParser;
  }

  /// Registers a combinator in the current scope.
  void registerCombinator(llvm::StringRef keyword, TypeInfo combinatorID,
                          DynParsingCombinator combinator) {
    if (!dynParser)
      return;
    dynParser->registerCombinator(keyword, combinatorID, combinator.parser,
                                  combinator.combinator);
  }

  template <typename T>
  inline void registerCombinator(llvm::StringRef keyword,
                                 DynParsingCombinator combinator) {
    registerCombinator(keyword, xblang::TypeInfo::get<T>(), combinator);
  }

  template <typename T>
  inline void registerCombinator(llvm::StringRef keyword,
                                 DynParsingCombinator::ParsingFn combinator) {
    registerCombinator(keyword, xblang::TypeInfo::get<T>(), {this, combinator});
  }

private:
  DynamicParser *dynParser{};
};

//===----------------------------------------------------------------------===//
// ParserMixin
//===----------------------------------------------------------------------===//
/// Utility class for defining parsers using the CRTP idiom.
template <typename Derived, typename Lex, typename... T>
class ParserMixin : public ParserBase, public T... {
public:
  using Base = ParserMixin<Derived, Lex, T...>;
  using Parser = Derived;
  using Lexer = Lex;
  using Token = typename Lexer::Token;
  using TokenID = typename Lexer::TokenID;
  using ParsingStatus = ::xblang::syntax::ParsingStatus;
  using ParserBase::getLoc;
  friend class xblang::SyntaxContext;

  ParserMixin(SourceManager &sourceManager, Lexer &lex)
      : ParserBase(sourceManager), lex(lex) {}

  Parser &getDerived() { return static_cast<Parser &>(*this); }

  const Parser &getDerived() const {
    return static_cast<const Parser &>(*this);
  }

  /// Returns the parsing id.
  static int getID() { return id; }

  /// Returns the current location of the lexer.
  SourceLocation getLoc() const { return lex.getLoc(); }

  /// Returns the current token being lexed.
  Token getTok() const { return lex.getTok(); }

  //// Consumes a new token.
  Token consume() { return lex.consume(); }

  /// Lexes a state without modifying the internal state.
  Token lexState(SourceState &state) const { return lex.lex(state); }

  /// Sets the state of the lexer.
  Parser &setState(SourceState &state) {
    lex(state);
    return getDerived();
  }

  /// Returns a parser guard.
  ParserGuard getGuard(bool emitError = true) {
    return ParserGuard(lex, emitError);
  }

  /// Parses a token using the dynamic parser.
  SourceState dynParse(DynParsingCombinator combinator, ParsingStatus &status,
                       bool emitErrors) {
    assert(combinator && "invalid combinator");
    SourceState state = lex.getState();
    state.restore(getTok().getLoc());
    combinator.combinator(combinator.parser, state, status, emitErrors);
    return state;
  }

protected:
  /// Production context.
  struct ProductionContext {
    ProductionContext &operator=(ManagedDiagnostic &&d) {
      if (diag)
        diag->abandon();
      diag = std::unique_ptr<ManagedDiagnostic>(
          new ManagedDiagnostic(std::move(d)));
      return *this;
    }

    llvm::StringRef errorSummary;
    const SourceLocation bLoc;
    ParserGuard guard;
    std::unique_ptr<ManagedDiagnostic> diag;
    uint32_t numErrors{};
  };

  typedef enum {
    CtrlPassthrough,
    CtrlProduction,
    CtrlSwitch,
    CtrlAny,
    CtrlZeroOrMore,
    CtrlNext,
    CtrlExit,
  } ControlState;

  /// Returns the pointer to the buffer.
  const char *getBuf() const { return getTok().getLoc().getLoc(); }

  /// Handle a control point.
  ControlState ctrlCheck(ProductionContext &ctx, ParserGuard &currentGuard,
                         ParserGuard &parentGuard, ControlState ctrl,
                         bool isNullable, bool isLocal, bool &emitErrors) {
    auto status = PureParsingStatus(currentGuard.getStatus());
    // When we are checking local rules (nullable switch or any `any`) we never
    // update the emitErrors arg.
    if (!isLocal)
      emitErrors = currentGuard.getEmitErrors();
    // If success then set the parent for success.
    if (currentGuard.isSuccess()) {
      parentGuard = success();
      currentGuard.release();
    } else if (isNullable || (ctrl == CtrlAny) || (ctrl == CtrlZeroOrMore)) {
      // If the status is not success but we are inside a never unsuccessful
      // construct, reset the state to the guard.
      currentGuard.reset();
      parentGuard = success();
      if (ctx.diag) {
        ctx.diag->abandon();
        ctx.diag.reset();
      }
    }
    // Get if there is an error.
    bool isError = currentGuard.getEmitErrors() && status.isAnError();
    // Emit any possible errors.
    if (isError && ctx.diag) {
      if (ctx.numErrors == 0)
        ctx.diag.reset();
      else if (ctx.diag) {
        ctx.diag->abandon();
        ctx.diag.reset();
      }
    }
    if (isError)
      ctx.numErrors++;
    // Handle control state.
    switch (ctrl) {
    case CtrlSwitch:
      // Return immediately if there is a non-recoverable error.
      if (isError || (status.isEmpty() && !isNullable)) {
        emitError(currentGuard.getLoc(),
                  ctx.errorSummary +
                      ", couldn't match any of the expected alternatives");
        parentGuard = error();
        return CtrlExit;
      }
      return isLocal ? CtrlNext : CtrlPassthrough;
    case CtrlAny:
      if (isError || status.isEmpty()) {
        emitError(currentGuard.getLoc(),
                  ctx.errorSummary +
                      ", couldn't match any of the alternatives");
        parentGuard = error();
        return CtrlExit;
      }
      return status.isSuccess() && isLocal ? CtrlNext : CtrlPassthrough;
    case CtrlZeroOrMore:
      if (status.isEmpty()) {
        currentGuard = success();
        parentGuard = success();
      }
      if (isError) {
        emitError(currentGuard.getLoc(),
                  ctx.errorSummary +
                      ", failed parsing a zero or more expression");
        return CtrlExit;
      }
      return CtrlPassthrough;
    default:
      return CtrlPassthrough;
    }
  }

  /// Returns the production context.
  ProductionContext getProductionContext(llvm::StringRef errorSummary,
                                         bool emitError) {
    return ProductionContext{errorSummary, getLoc(), getGuard(emitError),
                             nullptr};
  }

  /// Returns an empty status with the current location.
  ParsingStatus getStatus() const { return ParsingStatus::empty(getLoc()); }

  Lexer &lex;

private:
  static inline int id = -1;
};

/// Packrat parser context
class PackratContext {
private:
  using Key = std::pair<const char *, int>;
  using CachedValue = std::unique_ptr<ParsingStatus>;

public:
  template <typename Result, typename Lexer>
  ParseResult<Result> *getCache(Lexer &lexer, int key) {
    auto it = cache.find(Key(lexer.getBuf(), key));
    if (it == cache.end())
      return nullptr;
    auto result = static_cast<ParseCachedResult<Result> *>(it->second.get());
    lexer({result->getLoc(), result->getEnd()});
    return result;
  }

  void clearCache() { cache.clear(); }

protected:
  template <typename Result>
  inline ParseResult<Result>
  saveResult(std::optional<Result> &&value, PureParsingStatus::Status status,
             const SourceLocation &loc, const char *end, int prodId) {
    auto &cacheEntry = cache[Key(loc.getLoc(), prodId)];
    auto cachedVal = new ParseCachedResult<Result>(
        status, value ? std::move(*value) : Result{}, loc, end);
    cacheEntry = CachedValue(cachedVal);
    return *cachedVal;
  }

private:
  llvm::DenseMap<Key, CachedValue> cache{};
};
} // namespace syntax
} // namespace xblang

#endif
