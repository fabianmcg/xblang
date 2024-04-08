//===- SDTUtils.h - syntax-directed translator utils -------------*- C++-*-===//
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

#ifndef XBLANG_SYNTAX_UTILS_SDTUTILS_H
#define XBLANG_SYNTAX_UTILS_SDTUTILS_H

#include "xblang/Basic/Diagnostics.h"
#include "xblang/Basic/SourceState.h"
#include "llvm/ADT/StringRef.h"
#include <optional>

namespace xblang {
namespace syntaxgen {
class SDTLexerCommon {
public:
  struct Char {
    typedef enum : int32_t {
      Invalid = -1,
      UTF = 0,
      Control = 1,
    } State;

    Char(State state = Invalid, uint32_t c = 0) : state(state), character(c) {}

    static Char invalid() { return Char(Invalid, 0); }

    static Char utf(uint32_t c) { return Char(UTF, c); }

    static Char control(uint32_t c) { return Char(Control, c); }

    bool isInvalid() const { return state == Invalid; }

    bool isUTF() const { return state == UTF; }

    bool isControl() const { return state == Control; }

    int32_t state = Invalid;
    uint32_t character = 0;
  };

  /// Consumes an identifier from the source state.
  static llvm::StringRef consumeIdentifier(SourceState &state);
  /// Consumes an integer from the source state.
  static llvm::StringRef consumeNumber(SourceState &state);
  /// Consumes a hex integer from the source state.
  static llvm::StringRef consumeHexNumber(SourceState &state);
  /// Tries to consume a string literal from the source state.
  static std::optional<llvm::StringRef> consumeString(SourceState &state);
  /// Tries to consume a comment from the source state.
  static std::optional<llvm::StringRef> consumeComment(SourceState &state);
  /// Tries to consume a code section from the source state.
  static std::optional<llvm::StringRef> consumeCode(SourceState &state);
  /// Tries to consume a character from the source state.
  static Char consumeChar(SourceState &state, SourceLocation &loc,
                          std::string &error);

  /// Consume white-spaces from the source state.
  static void inline consumeSpaces(SourceState &state, uint32_t &character) {
    if (isspace(character))
      while (isspace(character = state.advance().get()))
        ;
  }

  /// Consume white-spaces from the source state.
  static void inline consumeSpaces(SourceState &state) {
    uint32_t character = *state;
    consumeSpaces(state, character);
  }
};
} // namespace syntaxgen
} // namespace xblang

#endif // XBLANG_SYNTAX_UTILS_SDTUTILS_H
