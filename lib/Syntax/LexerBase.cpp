//===- LexerBase.cpp -  ---------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the base XBLang lexer classes.
//
//===----------------------------------------------------------------------===//

#include "xblang/Syntax/LexerBase.h"

using namespace xblang;
using namespace xblang::syntax;

//===----------------------------------------------------------------------===//
// LexerGuard
//===----------------------------------------------------------------------===//
LexerGuard::LexerGuard(LexerBase &lexer)
    : lexer(&lexer), state(lexer.state), token(lexer.token) {}

LexerGuard::~LexerGuard() { reset(); }

void LexerGuard::reset() {
  if (lexer)
    (*lexer)(state, false).token = token;
}

void LexerGuard::release() { lexer = nullptr; }

//===----------------------------------------------------------------------===//
// LexerBase
//===----------------------------------------------------------------------===//
llvm::StringRef LexerBase::getSpelling(const SourceLocation &start,
                                       const SourceLocation &end) {
  assert(end.loc.getPointer() >= start.loc.getPointer() &&
         "invalid locations.");
  std::ptrdiff_t size = end.loc.getPointer() - start.loc.getPointer();
  return llvm::StringRef(start.loc.getPointer(), size);
}

llvm::StringRef LexerBase::getSpelling(const SourceLocation &start,
                                       const SourceLocation &end,
                                       size_t startOff, size_t endOff) {
  assert((end.loc.getPointer() - endOff) >=
             (start.loc.getPointer() + startOff) &&
         "invalid locations.");
  auto startLoc = (start.loc.getPointer() + startOff);
  std::ptrdiff_t size = (end.loc.getPointer() - endOff) - startLoc;
  return llvm::StringRef(startLoc, size);
}

LexerBase &LexerBase::operator()(const SourceState &state, bool consume) {
  this->state = state;
  if (consume)
    consumeToken();
  return *this;
}

LexerBase &LexerBase::operator()(const SourceLocation &location, bool consume) {
  if (state != location) {
    state.restore(location);
    if (consume)
      consumeToken();
  }
  return *this;
}

LexerBase &
LexerBase::operator()(const std::pair<SourceLocation, const char *> &other,
                      bool consume) {
  if (state != other.first) {
    state.restore(other.first, other.second);
    if (consume)
      consumeToken();
  }
  return *this;
}
