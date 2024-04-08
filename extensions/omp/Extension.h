//===- Extension.h - Declares the omp extension ------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the omp extension.
//
//===----------------------------------------------------------------------===//

#ifndef OMP_EXTENSION_H
#define OMP_EXTENSION_H

#include "omp/InitExtension.h"
#include "xblang/Lang/XBLang/Syntax/Lexer.h"
#include "xblang/Syntax/ParserBase.h"
#include "xblang/XLG/Builder.h"

namespace omp {
struct DataSharingInfo {
  typedef enum { Private, FirstPrivate, Shared } Kind;

  /// Token with name of the variable and source location.
  xblang::syntax::Token tok;
  /// Data sharing kind.
  Kind kind;

  /// Creates a private variable.
  static DataSharingInfo makePrivate(xblang::syntax::Token id) {
    return {id, Private};
  }

  /// Creates a firstprivate variable.
  static DataSharingInfo makeFirstPrivate(xblang::syntax::Token id) {
    return {id, FirstPrivate};
  }

  /// Creates a shared variable.
  static DataSharingInfo makeShared(xblang::syntax::Token id) {
    return {id, Shared};
  }
};
} // namespace omp

#include "omp/OMPParser.h.inc"

#endif // OMP_EXTENSION_H
