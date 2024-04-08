//===- XBGStmt.cpp - XBG stmt constructs ------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements XBG Stmt constructs.
//
//===----------------------------------------------------------------------===//

#include "xblang/Lang/XBLang/XLG/XBGStmt.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "xblang/Basic/Context.h"
#include "xblang/Lang/XBLang/XLG/XBGDialect.h"
#include "xblang/Lang/XBLang/XLG/XBGExpr.h"
#include "xblang/XLG/Concepts.h"

using namespace mlir;
using namespace xblang::xbg;

//===----------------------------------------------------------------------===//
// XBGStmt
//===----------------------------------------------------------------------===//

void XBGDialect::initializeStmt() {
  addOperations<
#define GET_OP_LIST
#include "xblang/Lang/XBLang/XLG/XBGStmt.cpp.inc"
      >();
  xblangContext->registerConstructs<
#define GET_OP_LIST
#include "xblang/Lang/XBLang/XLG/XBGStmt.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "xblang/Lang/XBLang/XLG/XBGStmt.cpp.inc"

#define GET_CLASS_DEF
#include "xblang/Lang/XBLang/XLG/XBGConceptStmt.cpp.inc"

namespace xblang {
namespace xbg {
#define GET_REGISTRATION_DEF
#include "xblang/Lang/XBLang/XLG/XBGConceptStmt.cpp.inc"
} // namespace xbg
} // namespace xblang
