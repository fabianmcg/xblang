//===- Syntax.cpp - Syntax ---------------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the syntax context as well as other functions and classes.
//
//===----------------------------------------------------------------------===//

#include "xblang/Syntax/SyntaxContext.h"

using namespace xblang;
using namespace xblang::syntax;

SyntaxContext::SyntaxContext(SourceManager &manager)
    : sourceManager(&manager) {}

DynParsingScope::DynParsingScope(DynamicParser &parser) {
  dynParser = &parser;
  prevScope = parser.currentScope;
  parser.currentScope = this;
}

DynParsingScope::~DynParsingScope() {
  if (!dynParser)
    return;
  while (!guard.empty()) {
    std::pair<Key, Value> prev = guard.back();
    guard.pop_back();
    if (!prev.second)
      dynParser->parserTable.erase(prev.first);
    else
      (dynParser->parserTable)[prev.first] = prev.second;
  }
  dynParser->currentScope = prevScope;
  dynParser = nullptr;
}

void DynParsingScope::insert(Key key, Value val) {
  auto &mapVal = (dynParser->parserTable)[key];
  guard.push_back({key, mapVal});
  mapVal = val;
}
