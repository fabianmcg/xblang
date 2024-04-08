//===- Syntax.h - Syntax context ---------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the syntax context.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_SYNTAX_SYNTAXCONTEXT_H
#define XBLANG_SYNTAX_SYNTAXCONTEXT_H

#include "xblang/Basic/SourceManager.h"
#include "xblang/Syntax/ParserBase.h"

namespace xblang {
class SyntaxContext {
public:
  SyntaxContext(SourceManager &manager);
  SyntaxContext(const SyntaxContext &) = delete;
  SyntaxContext &operator=(const SyntaxContext &) = delete;

  /// Returns a parser if it exists.
  template <typename Parser>
  Parser *getParser() {
    auto id = Parser::getID();
    if (id < 0)
      return nullptr;
    assert(id < static_cast<int>(parsers.size()) && "invalid parser ID");
    return static_cast<Parser *>(parsers[id].get());
  }

  /// Returns a Lexer if it exists.
  template <typename Lexer>
  Lexer *getLexer() {
    auto id = Lexer::getID();
    if (id < 0)
      return nullptr;
    assert(id < static_cast<int>(lexers.size()) && "invalid lexer ID");
    return static_cast<Lexer *>(lexers[id].get());
  }

  /// Returns or constructs a Lexer.
  template <typename Lexer, typename... Args>
  Lexer &getOrRegisterLexer(Args &&...args) {
    using Mixin = typename Lexer::Base;
    auto id = Lexer::getID();
    if (id < 0) {
      id = lexers.size();
      lexers.push_back(std::unique_ptr<syntax::LexerBase>(
          new Lexer(std::forward<Args>(args)..., *sourceManager)));
      static_cast<Mixin &>(*lexers.back()).id = id;
    }
    assert(id < static_cast<int>(lexers.size()) && "invalid lexer ID");
    return *static_cast<Lexer *>(lexers[id].get());
  }

  /// Returns or constructs a parser.
  template <typename Parser, typename... Args>
  Parser &getOrRegisterParser(Args &&...args) {
    using Mixin = typename Parser::Base;
    using Lexer = typename Parser::Lexer;
    auto id = Parser::getID();
    if (id < 0) {
      id = parsers.size();
      parsers.push_back(std::unique_ptr<syntax::ParserBase>(new Parser(
          std::forward<Args>(args)..., *sourceManager, *getLexer<Lexer>())));
      static_cast<Mixin &>(*parsers.back()).id = id;
    }
    assert(id < static_cast<int>(parsers.size()) && "invalid parser ID");
    Parser *parser = static_cast<Parser *>(parsers[id].get());
    return *parser;
  }

  /// Returns the dynamic parser owned by this context.
  syntax::DynamicParser &getDynParser() { return dynParser; }

private:
  SourceManager *sourceManager{};
  syntax::DynamicParser dynParser;
  std::vector<std::unique_ptr<syntax::LexerBase>> lexers{};
  std::vector<std::unique_ptr<syntax::ParserBase>> parsers{};
};
} // namespace xblang

#endif // XBLANG_SYNTAX_SYNTAXCONTEXT_H
