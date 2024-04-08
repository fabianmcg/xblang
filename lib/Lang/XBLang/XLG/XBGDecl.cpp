//===- XBGDecl.cpp - XBG decl constructs ------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements XBG Decl constructs.
//
//===----------------------------------------------------------------------===//

#include "xblang/Lang/XBLang/XLG/XBGDecl.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "xblang/Basic/Context.h"
#include "xblang/Lang/XBLang/XLG/XBGDialect.h"
#include "xblang/Sema/Sema.h"
#include "xblang/XLG/Concepts.h"

using namespace mlir;
using namespace xblang::xbg;

//===----------------------------------------------------------------------===//
// XBGDecl
//===----------------------------------------------------------------------===//

void XBGDialect::initializeDecl() {
  addOperations<
#define GET_OP_LIST
#include "xblang/Lang/XBLang/XLG/XBGDecl.cpp.inc"
      >();
  xblangContext->registerConstructs<
#define GET_OP_LIST
#include "xblang/Lang/XBLang/XLG/XBGDecl.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "xblang/Lang/XBLang/XLG/XBGDecl.cpp.inc"

#define GET_CLASS_DEF
#include "xblang/Lang/XBLang/XLG/XBGConceptDecl.cpp.inc"

namespace xblang {
namespace xbg {
#define GET_REGISTRATION_DEF
#include "xblang/Lang/XBLang/XLG/XBGConceptDecl.cpp.inc"
} // namespace xbg
} // namespace xblang

//===----------------------------------------------------------------------===//
// XBG FuncDecl
//===----------------------------------------------------------------------===//

::xblang::SymbolTableKind FuncDecl::getSymbolTableKind() {
  return SymbolTableKind::Ordered;
}

xblang::SymbolProperties FuncDecl::getSymbolProps() {
  return xblang::SymbolProperties::Overridable;
}
