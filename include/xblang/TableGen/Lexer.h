//===- Lexer.h - TableGen lexing classes  ------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares TableGen lexer related utilities.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_TABLEGEN_LEXER_H
#define XBLANG_TABLEGEN_LEXER_H

#include "xblang/TableGen/Record.h"
#include "llvm/TableGen/Error.h"

namespace llvm {
class Record;
} // namespace llvm

namespace xblang {
namespace tablegen {
//===----------------------------------------------------------------------===//
// Token
//===----------------------------------------------------------------------===//
// Wrapper class for the Token TableGen class.
class Token : public RecordMixin<Token> {
public:
  static constexpr std::string_view ClassType = "Token";
  using Base::Base;

  // Returns the name of the token.
  StringRef getName() const { return def->getValueAsString("identifier"); }

  // Returns the token aliases.
  std::vector<StringRef> getAliases() const {
    return def->getValueAsListOfStrings("aliases");
  }
};

//===----------------------------------------------------------------------===//
// TokenClass
//===----------------------------------------------------------------------===//
// Wrapper class for the Token TableGen class.
class TokenClass : public RecordMixin<TokenClass> {
public:
  static constexpr std::string_view ClassType = "TokenClass";
  using Base::Base;

  // Returns the name of the token.
  StringRef getName() const { return def->getValueAsString("identifier"); }

  // Returns the tokens.
  std::vector<Token> getTokens() const { return getDefList<Token>("tokens"); }
};

//===----------------------------------------------------------------------===//
// Keyword
//===----------------------------------------------------------------------===//
// Wrapper class for the Token TableGen class.
class Keyword : public RecordMixin<Keyword> {
public:
  static constexpr std::string_view ClassType = "Keyword";
  using Base::Base;

  // Returns the name of the token.
  StringRef getName() const { return def->getValueAsString("identifier"); }

  // Returns the keyword.
  StringRef getKeyword() const {
    std::vector<StringRef> aliases = def->getValueAsListOfStrings("aliases");
    if (aliases.empty())
      llvm::PrintFatalError(def, "keyword has no aliases");
    return aliases.front();
  }
};

//===----------------------------------------------------------------------===//
// Rule
//===----------------------------------------------------------------------===//
// Wrapper class for the Rule TableGen class.
class Rule : public RecordMixin<Rule> {
public:
  static constexpr std::string_view ClassType = "Rule";
  using Base::Base;

  // Returns the syntactical rule.
  StringRef getRule() const { return def->getValueAsString("rule"); }

  // Returns the code action.
  StringRef getAction() const { return def->getValueAsString("action"); }

  // Returns the rule as a token.
  Token getAsToken() const { return Token(def); }
};

//===----------------------------------------------------------------------===//
// Definition
//===----------------------------------------------------------------------===//
// Wrapper class for the Definition TableGen class.
class Definition : public RecordMixin<Definition> {
public:
  static constexpr std::string_view ClassType = "Definition";
  using Base::Base;

  // Returns the syntactical rule.
  StringRef getRule() const { return def->getValueAsString("expression"); }

  // Returns the definition's identifier.
  StringRef getIdentifier() const {
    return def->getValueAsString("identifier");
  }
};

//===----------------------------------------------------------------------===//
// DFA
//===----------------------------------------------------------------------===//
// Wrapper class for the DFA TableGen class.
class DFA : public RecordMixin<DFA> {
public:
  static constexpr std::string_view ClassType = "FiniteAutomata";
  using Base::Base;

  /// Returns the identifier of the DFA.
  StringRef getIdentifier() const { return def->getValueAsString("name"); }

  /// Returns whether the DFA will ignore white spaces.
  bool getIgnoreWhitespace() const {
    return def->getValueAsBit("ignoreWhitespace");
  }

  /// Returns whether the DFA will be enclosed in a loop.
  bool getLoop() const { return def->getValueAsBit("loop"); }

  /// Returns the default action to perform on error.
  StringRef getErrorAction() const {
    return def->getValueAsString("errorAction");
  }

  /// Returns a list of the DFA definitions.
  std::vector<Definition> getDefinitions() const {
    return getDefList<Definition>("definitions");
  }

  /// Returns a list of the DFA rules.
  std::vector<Rule> getRules() const { return getDefList<Rule>("rules"); }
};

//===----------------------------------------------------------------------===//
// Lexer
//===----------------------------------------------------------------------===//
// Wrapper class for the Lexer TableGen class.
class Lexer : public RecordMixin<Lexer> {
public:
  static constexpr std::string_view ClassType = "Lexer";
  using Base::Base;

  /// Returns the identifier of the Lexer.
  llvm::StringRef getIdentifier() const {
    return def->getValueAsString("name");
  }

  /// Returns the namespace of the Lexer.
  llvm::StringRef getCppNamespace() const {
    return def->getValueAsString("cppNamespace");
  }

  /// Returns the `implement` bit-field.
  bool getImplement() const { return def->getValueAsBit("implement"); }

  /// Returns the lexing DFAs field.
  std::vector<DFA> getDFAutomatons() const {
    return getDefList<DFA>("automatons");
  }

  /// Returns the lexing tokens.
  std::vector<Token> getTokens() const { return getDefList<Token>("tokens"); }

  /// Returns the lexing token classes.
  std::vector<TokenClass> getTokenClasses() const {
    return getDefList<TokenClass>("tokenClasses");
  }
};
} // namespace tablegen
} // namespace xblang

#endif // XBLANG_TABLEGEN_LEXER_H
