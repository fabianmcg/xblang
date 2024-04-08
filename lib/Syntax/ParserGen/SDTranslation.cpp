//===- SDTranslation.cpp - Parser syntax-directed translator  -----*-
// C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the syntax-directed translator for the Parser IR.
//
//===----------------------------------------------------------------------===//

#include "xblang/Syntax/ParserGen/SDTranslation.h"
#include "mlir/IR/Builders.h"
#include "xblang/Support/Format.h"
#include "xblang/Syntax/IR/SyntaxDialect.h"
#include "xblang/Syntax/LexerBase.h"

using namespace xblang;
using namespace xblang::syntax;
using namespace xblang::syntaxgen::parser;

//===----------------------------------------------------------------------===//
// TokenMap
//===----------------------------------------------------------------------===//

TerminalMap::TerminalMap(tablegen::Lexer lexer, llvm::StringRef unspecifiedKey)
    : lexer(lexer), unspecifiedKey(unspecifiedKey) {
  auto addConstruct = [&](llvm::StringRef str, RecordTy rec) {
    tokenMap[str] = rec;
    if (!defaultConstruct && str == unspecifiedKey)
      defaultConstruct = rec;
  };
  for (auto tok : lexer.getTokens()) {
    addConstruct(tok.getName(), &tok.getDef());
    for (auto alias : tok.getAliases())
      addConstruct(alias, &tok.getDef());
  }
  for (auto tok : lexer.getTokenClasses())
    addConstruct(tok.getName(), &tok.getDef());
}

//===----------------------------------------------------------------------===//
// SDTLexer
//===----------------------------------------------------------------------===//
llvm::StringRef SDTLexer::toString(TokenID value) {
  switch (value) {
#define TERMINAL(_0, ...)                                                      \
  case TokenID::_0:                                                            \
    return #_0;
#include "xblang/Syntax/ParserGen/Tokens.inc"
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
    } /* Lex keyword or identifier */ else if (isalpha(character) ||
                                               character == '_') {
      spelling = consumeIdentifier(state);
      tok = TokenID::Identifier;
      break;
    } /* Lex string */ else if (character == '"' || character == '\'') {
      std::optional<llvm::StringRef> value = consumeString(state);
      if (!value) {
        emitError(beginLoc, "the string was never closed");
        tok = TokenID::Invalid;
        break;
      }
      spelling = *value;
      tok = TokenID::String;
      break;
    } /* Lex code */ else if (character == '{' && state.at(1) == '{') {
      std::optional<llvm::StringRef> value = consumeCode(state);
      if (!value) {
        emitError(beginLoc, "the code literal was never closed");
        tok = TokenID::Invalid;
        break;
      }
      spelling = *value;
      tok = TokenID::Code;
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
#include "xblang/Syntax/ParserGen/Tokens.inc"
              .Default(TokenID::Invalid);
  if (tok == TokenID::Invalid) {
    switch (state.get()) {
#define CHAR_PUNCTUATION(_0, _1)                                               \
  case _1:                                                                     \
    tok = TokenID::_0;                                                         \
    break;
#include "xblang/Syntax/ParserGen/Tokens.inc"
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
SDTranslator::SDTranslator(SourceManager &sourceManager, TerminalMap &tokMap,
                           mlir::SymbolTable &symTable)
    : sourceManager(sourceManager), tokMap(tokMap), lexer(sourceManager),
      symTable(symTable) {}

SDTranslator::~SDTranslator() = default;

bool SDTranslator::match(Lexer::TokenID tok, Lexer::TokenID expected) {
  if (tok != expected) {
    sourceManager.emitError(lexer.getLoc(),
                            "expected: '" + lexer.toString(expected) +
                                "', but got: '" + lexer.toString(tok) + "'");
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

ParseResult<mlir::Operation *>
SDTranslator::parseProduction(Source *source, llvm::StringRef id,
                              mlir::Block &block) {
  assert(source && "null-source");
  state = source->getState();
  if (state.isValid())
    lexer(state);
  else
    return ParseResult<mlir::Operation *>::error();
  arguments.clear();
  auto bLoc = lexer.getLoc();
  mlir::OpBuilder builder(symTable.getOp()->getContext());
  auto def = builder.create<RuleOp>(sourceManager.getLoc(bLoc), id);
  auto symName = symTable.insert(def, block.end());
  if (symName.getValue() != id) {
    def.emitError("redefinition of symbol: " + id);
    return ParseResult<mlir::Operation *>::error();
  }
  def.getBodyRegion().emplaceBlock();
  builder.setInsertionPoint(def.getBody(0), def.getBody(0)->end());
  mlir::Value expr;
  while (true) {
    auto tok = lexer.getTok();
    if (tok == Lexer::Invalid)
      return ParseResult<mlir::Operation *>::error(tok.getLoc());
    size_t sid = 0;
    auto rule = parseExpr(nullptr, true, builder, sid);
    if (!rule.isSuccess())
      return ParseOp::error(rule.getLoc());
    expr = expr ? builder.create<OrOp>(sourceManager.getLoc(rule.getLoc()),
                                       expr, rule.get())
                : rule.get();
    tok = lexer.getTok();
    if (tok == Lexer::Semicolon || tok == Lexer::EndOfFile) {
      lexer.consume();
      break;
    }
    if (tok == Lexer::BOr)
      lexer.consume();
  }
  if (!expr)
    return ParseResult<mlir::Operation *>::error();
  builder.create<ReturnOp>(sourceManager.getLoc(bLoc), expr);
  return ParseOp::success(def, bLoc);
}

syntax::ParseResult<mlir::Operation *>
SDTranslator::parseMacro(Source *source,
                         const std::vector<llvm::StringRef> &args,
                         llvm::StringRef id, mlir::Block &block) {
  assert(source && "null-source");
  state = source->getState();
  if (state.isValid())
    lexer(state);
  else
    return ParseResult<mlir::Operation *>::error();
  auto bLoc = lexer.getLoc();
  mlir::OpBuilder builder(symTable.getOp()->getContext());
  auto def =
      builder.create<MacroOp>(sourceManager.getLoc(bLoc), id, args.size());
  auto symName = symTable.insert(def, block.end());
  if (symName.getValue() != id) {
    def.emitError("redefinition of symbol: " + id);
    return ParseResult<mlir::Operation *>::error();
  }
  builder.setInsertionPoint(def.getBody(0), def.getBody(0)->end());
  arguments.clear();
  for (auto [i, arg] : llvm::enumerate(args))
    arguments[arg] = def.getArgument(i);
  size_t sid = 0;
  auto rule = parseExpr(nullptr, false, builder, sid);
  if (!rule.isSuccess())
    return ParseOp::error(rule.getLoc());
  builder.create<ReturnOp>(sourceManager.getLoc(bLoc), rule.get());
  return ParseOp::success(def, bLoc);
}

ParseResult<mlir::Value> SDTranslator::parseExpr(mlir::Value lhs,
                                                 bool parseAsRule,
                                                 mlir::OpBuilder &builder,
                                                 size_t &sid) {
  mlir::Value expr;
  auto bLoc = lexer.getLoc();
  StringRef preCodeAction = "", postCodeAction = "";
  auto tok = lexer.getTok();
  if (tok == Lexer::Code) {
    preCodeAction = tok.getSpelling().drop_back(2).drop_front(2);
    lexer.consume();
  }
  while (true) {
    tok = lexer.getTok();
    if (tok == Lexer::Invalid)
      return ParseValue::error(tok.getLoc());
    if (tok == Lexer::EndOfFile || tok == Lexer::BOr ||
        tok == Lexer::Semicolon || tok == Lexer::RParen ||
        tok == Lexer::Comma || tok == Lexer::Code)
      break;
    auto subExpr = parseTopExpr(builder, sid);
    if (!subExpr.isSuccess())
      return subExpr;
    expr = expr ? builder.create<AndOp>(sourceManager.getLoc(tok.getLoc()),
                                        expr, subExpr.get())
                : subExpr.get();
  }
  if (!expr) {
    sourceManager.emitError(bLoc, "empty expressions are not valid.");
    return ParseValue::error(bLoc);
  }
  tok = lexer.getTok();
  if (tok == Lexer::Code) {
    postCodeAction = tok.getSpelling().drop_back(2).drop_front(2);
    lexer.consume();
    tok = lexer.getTok();
    // Check for syntax errors.
    if (tok != Lexer::Semicolon && tok != Lexer::EndOfFile &&
        tok != Lexer::BOr && tok != Lexer::RParen) {
      sourceManager.emitError(
          tok.getLoc(),
          "the production was terminated unexpectedly by a code literal");
      return ParseValue::error(tok.getLoc());
    }
  }
  // Attach a code action.
  if (!preCodeAction.empty() || !postCodeAction.empty())
    expr = builder.create<MetadataOp>(
        expr.getLoc(), expr, nullptr,
        builder.getAttr<CodeActionAttr>(preCodeAction, postCodeAction));
  if (lhs)
    expr = builder.create<OrOp>(lhs.getLoc(), lhs, expr);
  if (lexer.getTok() == Lexer::BOr) {
    if (!parseAsRule) {
      lexer.consume();
      size_t tid = sid;
      return parseExpr(expr, false, builder, tid);
    }
  }
  return ParseValue::success(std::move(expr), bLoc);
}

ParseResult<mlir::Value> SDTranslator::parseTopExpr(mlir::OpBuilder &builder,
                                                    size_t &sid) {
  auto tok = lexer.getTok();
  ParseValue expr;
  switch (tok.getTok()) {
  case Lexer::LParen:
    expr = parseParenExpr(builder, sid);
    if (!expr.isSuccess())
      return expr;
    break;
  case Lexer::String:
    expr = parseTerminalExpr(builder, sid);
    if (!expr.isSuccess())
      return expr;
    break;
  case Lexer::Identifier:
    expr = parseNonTerminalExpr(builder, sid);
    if (!expr.isSuccess())
      return expr;
    break;
  case Lexer::At:
    expr = parseCallExpr(builder, sid);
    if (!expr.isSuccess())
      return expr;
    break;
  case Lexer::Number:
    expr = parseDirective(builder, sid);
    if (!expr.isSuccess())
      return expr;
    break;
  case Lexer::EmptyString: {
    expr = ParseValue::success(
        builder.create<EmptyStringOp>(sourceManager.getLoc(tok.getLoc())),
        tok.getLoc());
    lexer.consume();
    break;
  }
  default:
    sourceManager.emitError(tok.getLoc(), "invalid top expression");
    return ParseValue::error(tok.getLoc());
  }
  while (true) {
    tok = lexer.getTok();
    switch (tok.getTok()) {
    case Lexer::Multiply: {
      lexer.consume();
      expr.get() =
          builder.create<ZeroOrMoreOp>(expr.get().getLoc(), expr.get());
      continue;
    }
    case Lexer::Plus: {
      lexer.consume();
      auto zeroOrMore = builder.create<ZeroOrMoreOp>(expr.get().getLoc(),
                                                     expr.get(), nullptr);
      expr.get() =
          builder.create<AndOp>(expr.get().getLoc(), expr.get(), zeroOrMore);
      continue;
    }
    case Lexer::Question: {
      lexer.consume();
      expr.get() = builder.create<OrOp>(
          expr.get().getLoc(), expr.get(),
          builder.create<EmptyStringOp>(expr.get().getLoc()));
      continue;
    }
    default:
      break;
    }
    break;
  }
  return expr;
}

ParseResult<mlir::Value> SDTranslator::parseParenExpr(mlir::OpBuilder &builder,
                                                      size_t &sid) {
  auto lparen = matchAndConsume(Lexer::LParen);
  if (!lparen.first)
    return ParseValue::error(lparen.second);
  auto expr = parseExpr(nullptr, false, builder, sid);
  if (!expr.isSuccess()) {
    sourceManager.emitError(expr.getLoc(), "expected a non empty expression");
    return ParseValue::error(expr.getLoc());
  }
  auto rparen = matchAndConsume(Lexer::RParen);
  if (!rparen.first)
    return ParseValue::error(rparen.second);
  return ParseValue::success(std::move(expr.get()), lparen.second);
}

ParseResult<mlir::Value>
SDTranslator::parseTerminalExpr(mlir::OpBuilder &builder, size_t &sid) {
  auto tok = lexer.getTok();
  auto res = matchAndConsume(Lexer::String);
  if (!res.first)
    return ParseValue::error(res.second);
  auto name = ParseResult<mlir::FlatSymbolRefAttr>::empty();
  if (lexer.getTok() == Lexer::Colon)
    name = parseNameAttr(builder);
  if (name.isAnError())
    return ParseValue::error(name.getLoc());
  auto spelling = tok.getSpelling().drop_front(1).drop_back(1);
  // Find the terminal in the terminal map.
  auto record = tokMap.get(spelling);
  bool isUnspecified = false, isPresent = record != nullptr;
  LexTerminalAttr terminalAttr;
  if (record == nullptr)
    isUnspecified = (record = tokMap.getDefaultConstruct()) != nullptr;
  // Only tokens can be specified using quotes.
  if (auto tok = tablegen::Token::castOrNull(record))
    terminalAttr = builder.getAttr<LexTerminalAttr>(
        builder.getAttr<mlir::FlatSymbolRefAttr>(tok->getName()),
        isUnspecified ? LexTerminalKind::Unspecified : LexTerminalKind::Token,
        isUnspecified ? spelling : "");
  // Emit an error if the token couldn't be found or there's no unspecified
  // construct.
  if (!terminalAttr) {
    if (isPresent)
      sourceManager.emitError(tok.getLoc(),
                              "token classes should appear without quotes");
    else
      sourceManager.emitError(tok.getLoc(), "terminal couldn't be resolved");
    return ParseValue::error(res.second);
  }
  // Check for syntax errors if we are inside a macro.
  if (arguments.size() > 0 && !name.isSuccess()) {
    name = ParseResult<mlir::FlatSymbolRefAttr>::success(
        builder.getAttr<mlir::FlatSymbolRefAttr>(fmt("__{0}", symId++)));
  }
  // Check for code actions.
  CodeActionAttr codeAction;
  if (auto tok = lexer.getTok(); tok == Lexer::Code) {
    codeAction = builder.getAttr<CodeActionAttr>(
        "", tok.getSpelling().drop_back(2).drop_front(2));
    lexer.consume();
  }
  // Increment the symbol ID counter.
  mlir::FlatSymbolRefAttr nameAttr =
      name.isSuccess()
          ? name.get()
          : builder.getAttr<mlir::FlatSymbolRefAttr>(fmt("_{0}", sid));
  ++sid;
  auto terminal = builder.create<TerminalOp>(sourceManager.getLoc(tok.getLoc()),
                                             terminalAttr);
  auto action = builder.create<MetadataOp>(terminal.getLoc(), terminal,
                                           nameAttr, codeAction);
  return ParseValue::success(action, tok.getLoc());
}

ParseResult<mlir::Value>
SDTranslator::parseNonTerminalExpr(mlir::OpBuilder &builder, size_t &sid) {
  auto tok = lexer.getTok();
  auto res = matchAndConsume(Lexer::Identifier);
  if (!res.first)
    return ParseValue::error(res.second);
  auto name = ParseResult<mlir::FlatSymbolRefAttr>::empty();
  if (lexer.getTok() == Lexer::Colon)
    name = parseNameAttr(builder);
  if (name.isAnError())
    return ParseValue::error(name.getLoc());
  // Increment the symbol ID counter.
  mlir::FlatSymbolRefAttr nameAttr =
      name.isSuccess()
          ? name.get()
          : builder.getAttr<mlir::FlatSymbolRefAttr>(fmt("_{0}", sid));
  ++sid;
  // Check if it's a macro argument
  auto it = arguments.find(tok.getSpelling());
  if (it != arguments.end())
    return ParseValue::success(mlir::Value(it->second), tok.getLoc());
  // Check for syntax errors if we are inside a macro.
  if (arguments.size() > 0 && !name.isSuccess()) {
    sourceManager.emitError(
        res.second, "all nonterminals inside a macro must provide a name.");
    return ParseValue::error(res.second);
  }
  // Check for code actions.
  CodeActionAttr codeAction;
  if (auto tok = lexer.getTok(); tok == Lexer::Code) {
    codeAction = builder.getAttr<CodeActionAttr>(
        "", tok.getSpelling().drop_back(2).drop_front(2));
    lexer.consume();
  }
  // Try to find a terminal in the terminal map.
  auto record = tokMap.get(tok.getSpelling());
  if (record) {
    // Emit the found terminal.
    LexTerminalAttr terminalAttr;
    if (auto tok = tablegen::Token::castOrNull(record))
      terminalAttr = builder.getAttr<LexTerminalAttr>(
          builder.getAttr<mlir::FlatSymbolRefAttr>(tok->getName()),
          LexTerminalKind::Token, "");
    else if (auto tokClass = tablegen::TokenClass::castOrNull(record))
      terminalAttr = builder.getAttr<LexTerminalAttr>(
          builder.getAttr<mlir::FlatSymbolRefAttr>(tokClass->getName()),
          LexTerminalKind::Class, "");
    auto terminal = builder.create<TerminalOp>(
        sourceManager.getLoc(tok.getLoc()), terminalAttr);
    auto action = builder.create<MetadataOp>(terminal.getLoc(), terminal,
                                             nameAttr, codeAction);
    return ParseValue::success(action, tok.getLoc());
  }
  // Emit the non-terminal.
  auto nt = builder.create<NonTerminalOp>(sourceManager.getLoc(tok.getLoc()),
                                          tok.getSpelling(), nullptr);
  auto action =
      builder.create<MetadataOp>(nt.getLoc(), nt, nameAttr, codeAction);
  return ParseValue::success(action, tok.getLoc());
}

ParseResult<mlir::Value> SDTranslator::parseCallExpr(mlir::OpBuilder &builder,
                                                     size_t &sid) {
  auto atSign = matchAndConsume(Lexer::At);
  if (!atSign.first)
    return ParseValue::error(atSign.second);
  auto tok = lexer.getTok();
  auto res = matchAndConsume(Lexer::Identifier);
  if (!res.first)
    return ParseValue::error(res.second);
  auto macro = symTable.lookup<MacroOp>(tok.getSpelling());
  if (!macro) {
    sourceManager.emitError(res.second,
                            "invalid reference to inexistent symbol: " +
                                tok.getSpelling());
    return ParseValue::error(res.second);
  }
  auto lparen = matchAndConsume(Lexer::LParen);
  if (!lparen.first)
    return ParseValue::error(lparen.second);
  SmallVector<mlir::Value> args;
  while (true) {
    auto tok = lexer.getTok();
    if (tok == Lexer::RParen || tok == Lexer::Invalid ||
        tok == Lexer::EndOfFile)
      break;
    auto expr = parseExpr(nullptr, false, builder, sid);
    if (!expr.isSuccess()) {
      sourceManager.emitError(expr.getLoc(), "expected a non empty expression");
      return ParseValue::error(expr.getLoc());
    }
    args.push_back(expr.get());
    if (lexer.getTok() == Lexer::Comma)
      lexer.consume();
  }
  auto rparen = matchAndConsume(Lexer::RParen);
  if (!rparen.first)
    return ParseValue::error(rparen.second);
  if (args.size() != macro.getNumArguments()) {
    sourceManager.emitError(res.second,
                            fmt("macro expected {0} arguments but got: {1}",
                                macro.getNumArguments(), args.size()));
    return ParseValue::error(res.second);
  }
  auto expr = builder.create<CallOp>(sourceManager.getLoc(atSign.second),
                                     tok.getSpelling(), args);
  return ParseValue::success(std::move(expr), atSign.second);
}

ParseResult<mlir::Value> SDTranslator::parseDirective(mlir::OpBuilder &builder,
                                                      size_t &sid) {
  auto sign = matchAndConsume(Lexer::Number);
  if (!sign.first)
    return ParseValue::error(sign.second);
  auto tok = lexer.getTok();
  auto res = matchAndConsume(Lexer::Identifier);
  if (!res.first)
    return ParseValue::error(res.second);
  Value expr{};
  // Check for code actions.
  llvm::StringRef directive = tok.getSpelling();
  if (directive == "dyn_kw" || directive == "dyn") {
    res = matchAndConsume(Lexer::LParen);
    if (!res.first)
      return ParseValue::error(res.second);
    tok = lexer.getTok();
    auto res = matchAndConsume(Lexer::String);
    if (!res.first)
      return ParseValue::error(res.second);
    auto spelling = tok.getSpelling().drop_front(1).drop_back(1);
    res = matchAndConsume(Lexer::RParen);
    if (!res.first)
      return ParseValue::error(res.second);
    auto name = ParseResult<mlir::FlatSymbolRefAttr>::empty();
    if (lexer.getTok() == Lexer::Colon)
      name = parseNameAttr(builder);
    // Increment the symbol ID counter.
    mlir::FlatSymbolRefAttr nameAttr =
        name.isSuccess()
            ? name.get()
            : builder.getAttr<mlir::FlatSymbolRefAttr>(fmt("_{0}", sid));
    ++sid;
    if (name.isAnError())
      return ParseValue::error(name.getLoc());
    if (directive == "dyn_kw")
      expr = builder.create<TerminalOp>(
          sourceManager.getLoc(tok.getLoc()),
          builder.getAttr<LexTerminalAttr>(
              builder.getAttr<mlir::FlatSymbolRefAttr>(directive),
              LexTerminalKind::Dynamic, spelling));
    else
      expr = builder.create<NonTerminalOp>(sourceManager.getLoc(tok.getLoc()),
                                           "_$dyn",
                                           builder.getStringAttr(spelling));
    CodeActionAttr codeAction;
    if (auto tok = lexer.getTok(); tok == Lexer::Code) {
      codeAction = builder.getAttr<CodeActionAttr>(
          "", tok.getSpelling().drop_back(2).drop_front(2));
      lexer.consume();
    }
    expr =
        builder.create<MetadataOp>(expr.getLoc(), expr, nameAttr, codeAction);
  } else {
    sourceManager.emitError(res.second, "unknown directive");
    return ParseValue::error(res.second);
  }
  return ParseValue::success(std::move(expr), sign.second);
}

ParseResult<mlir::FlatSymbolRefAttr>
SDTranslator::parseNameAttr(mlir::OpBuilder &builder) {
  auto bLoc = lexer.getLoc();
  auto res = matchAndConsume(Lexer::Colon);
  if (!res.first)
    return ParseResult<mlir::FlatSymbolRefAttr>::error(res.second);
  res = matchAndConsume(Lexer::Dollar);
  if (!res.first)
    return ParseResult<mlir::FlatSymbolRefAttr>::error(res.second);
  auto tok = lexer.getTok();
  res = matchAndConsume(Lexer::Identifier);
  if (!res.first)
    return ParseResult<mlir::FlatSymbolRefAttr>::error(res.second);
  return ParseResult<mlir::FlatSymbolRefAttr>::success(
      builder.getAttr<mlir::FlatSymbolRefAttr>(tok.getSpelling()), bLoc);
}
