//===- SDTUtils.cpp - syntax-directed translator utils -----------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares syntax-directed translator utilities.
//
//===----------------------------------------------------------------------===//

#include "xblang/Syntax/Utils/SDTUtils.h"

using namespace xblang;
using namespace xblang::syntaxgen;

llvm::StringRef SDTLexerCommon::consumeIdentifier(SourceState &state) {
  auto buffer = state.begin();
  auto validChar = [](char c) { return isalnum(c) || c == '_'; };
  while (validChar(state.advance().get()))
    ;
  return llvm::StringRef(buffer, static_cast<size_t>(state.begin() - buffer));
}

llvm::StringRef SDTLexerCommon::consumeNumber(SourceState &state) {
  auto begin = state.begin();
  while (isdigit(state.advance().get()))
    ;
  return llvm::StringRef(begin, static_cast<size_t>(state.begin() - begin));
}

llvm::StringRef SDTLexerCommon::consumeHexNumber(SourceState &state) {
  auto begin = state.begin();
  while (isxdigit(state.advance().get()))
    ;
  return llvm::StringRef(begin, static_cast<size_t>(state.begin() - begin));
}

std::optional<llvm::StringRef>
SDTLexerCommon::consumeString(SourceState &state) {
  auto buffer = state.begin();
  auto quote = *state;
  bool finished = false;
  while (state.advance().get()) {
    if (*state == '\\' && state.at(1) == quote) {
      ++state;
      continue;
    }
    if (*state == quote) {
      ++state;
      finished = true;
      break;
    }
  }
  if (!finished)
    return std::nullopt;
  return llvm::StringRef(buffer, static_cast<size_t>(state.begin() - buffer));
}

std::optional<llvm::StringRef>
SDTLexerCommon::consumeComment(SourceState &state) {
  auto buffer = state.begin();
  if (state.at(1) == '/') {
    auto validChar = [](char c) { return c && !(c == '\r' || c == '\n'); };
    while (validChar(state.advance().get()))
      ;
  } else {
    ++state;
    bool finished = false;
    while (state.advance().get()) {
      if (*state == '*' && state.at(1) == '/') {
        ++state;
        ++state;
        finished = true;
        break;
      }
    }
    if (!finished)
      return std::nullopt;
  }
  return llvm::StringRef(buffer, static_cast<size_t>(state.begin() - buffer));
}

std::optional<llvm::StringRef> SDTLexerCommon::consumeCode(SourceState &state) {
  auto buffer = state.begin();
  ++state;
  bool finished = false;
  while (state.advance().get())
    if (*state == '}' && state.at(1) == '}') {
      ++state;
      ++state;
      finished = true;
      break;
    }
  if (!finished)
    return std::nullopt;
  return llvm::StringRef(buffer, static_cast<size_t>(state.begin() - buffer));
}

SDTLexerCommon::Char SDTLexerCommon::consumeChar(SourceState &state,
                                                 SourceLocation &loc,
                                                 std::string &error) {
  uint32_t character = 0;
  auto buffer = state.begin();
  state.advance();
  bool escapedCharacter = false;
  if (buffer[0] == '\\') {
    switch (state.at(0)) {
    case 'n': {
      character = '\n';
      state.advance();
      break;
    }
    case 't': {
      character = '\t';
      state.advance();
      break;
    }
    case ' ': {
      character = ' ';
      state.advance();
      break;
    }
    case '-': {
      character = '-';
      state.advance();
      break;
    }
    case '\\': {
      character = '\\';
      state.advance();
      break;
    }
    case '/': {
      character = '/';
      state.advance();
      break;
    }
    case ']': {
      character = ']';
      state.advance();
      break;
    }
    case '\'': {
      character = '\'';
      state.advance();
      break;
    }
    case 'u':
    case 'U': {
      loc = state.getLoc();
      state.advance();
      if (*state != '+') {
        error = "invalid unicode character";
        return Char::invalid();
      }
      state.advance();
      character = 0;
      auto spelling = consumeHexNumber(state);
      bool err = spelling.getAsInteger(16, character);
      if (err) {
        error = "invalid hex number";
        return Char::invalid();
      }
      break;
    }
    default:
      error = "invalid escaped character";
      return Char::invalid();
    }
    escapedCharacter = true;
  }
  if (!escapedCharacter)
    character = buffer[0];
  if (buffer[0] == ']' || buffer[0] == '-' || buffer[0] == '\'')
    return Char::control(buffer[0]);
  return Char::utf(character);
}
