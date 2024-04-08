//===- Parser.h - TableGen parsing classes  ----------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares TableGen parser related utilities.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_TABLEGEN_PARSER_H
#define XBLANG_TABLEGEN_PARSER_H

#include "xblang/TableGen/Common.h"
#include "xblang/TableGen/Lexer.h"
#include "xblang/TableGen/Record.h"

namespace llvm {
class Record;
} // namespace llvm

namespace xblang {
namespace tablegen {
//===----------------------------------------------------------------------===//
// Production
//===----------------------------------------------------------------------===//
/// Wrapper class for the Production TableGen class.
class Production : public RecordMixin<Production> {
public:
  static constexpr std::string_view ClassType = "Production";
  using Base::Base;

  /// Returns the syntactical rule.
  StringRef getRule() const { return def->getValueAsString("rule"); }

  /// Returns the identifier of the production.
  StringRef getIdentifier() const {
    return def->getValueAsString("identifier");
  }

  /// Returns the return type of the production.
  StringRef getReturnType() const {
    return def->getValueAsString("returnType");
  }

  /// Returns the `implement` bit-field.
  bool getImplement() const { return def->getValueAsBit("implement"); }

  /// Returns the `arguments` field.
  ParameterList getArguments() const {
    return ParameterList::get(def->getValueAsDag("arguments"));
  }
};

//===----------------------------------------------------------------------===//
// ParserMacro
//===----------------------------------------------------------------------===//
/// Wrapper class for the ParserMacro TableGen class.
class ParserMacro : public RecordMixin<ParserMacro> {
public:
  static constexpr std::string_view ClassType = "ParserMacro";
  using Base::Base;

  /// Returns the macro expression.
  StringRef getExpr() const { return def->getValueAsString("expression"); }

  /// Returns the identifier of the production.
  StringRef getIdentifier() const {
    return def->getValueAsString("identifier");
  }

  /// Returns the arguments names.
  std::vector<StringRef> getArgs() const {
    return def->getValueAsListOfStrings("arguments");
  }
};

//===----------------------------------------------------------------------===//
// Parser
//===----------------------------------------------------------------------===//
/// Wrapper class for the Parser TableGen class.
class Parser : public RecordMixin<Parser> {
public:
  static constexpr std::string_view ClassType = "Parser";
  using Base::Base;

  /// Returns the identifier of the parser.
  llvm::StringRef getIdentifier() const {
    return def->getValueAsString("name");
  }

  /// Returns the namespace of the parser.
  llvm::StringRef getCppNamespace() const {
    return def->getValueAsString("cppNamespace");
  }

  /// Returns the start symbol of the parser.
  llvm::StringRef getStartSymbol() const {
    return def->getValueAsString("startSymbol");
  }

  /// Returns the lexer.
  Lexer getLexer() const { return getDef<Lexer>("lexer"); }

  /// Returns the `implement` bit-field.
  bool getImplement() const { return def->getValueAsBit("implement"); }

  /// Returns a list of the parser productions.
  std::vector<Production> getProductions() const {
    return getDefList<Production>("productions");
  }

  /// Returns a list of the parser macros.
  std::vector<ParserMacro> getMacros() const {
    return getDefList<ParserMacro>("macros");
  }

  /// Returns the default token to used when building the parser.
  llvm::StringRef getDefaultToken() const {
    return def->getValueAsString("defaultToken");
  }

  /// Returns the traits list.
  llvm::SmallVector<Trait> getTraits() const {
    assert(def && "null record.");
    llvm::ListInit *list = def->getValueAsListInit("traits");
    llvm::SmallVector<Trait> defs;
    for (llvm::Init *init : list->getValues())
      defs.push_back(Trait::create(init));
    return defs;
  }
};
} // namespace tablegen
} // namespace xblang

#endif // XBLANG_TABLEGEN_PARSER_H
