//===- XBGDialect.cpp - XBG dialect -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the XBG dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "xblang/Basic/ContextDialect.h"
#include "xblang/Dialect/XBLang/IR/Dialect.h"
#include "xblang/Lang/XBLang/XLG/XBGDialect.h"
#include "xblang/Support/Format.h"
#include "xblang/XLG/IR/XLGDialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace xblang::xbg;

//===----------------------------------------------------------------------===//
// XBGDialect
//===----------------------------------------------------------------------===

namespace {
class XGBContextInterface : public xblang::XBContextDialectInterface {
public:
  using xblang::XBContextDialectInterface::XBContextDialectInterface;

  xblang::XBContext *geXBContext() override {
    return cast<XBGDialect>(getDialect())->getXBContext();
  }
};
} // namespace

void XBGDialect::initialize() {
  XBContextDialect *xbCtx = getContext()->getLoadedDialect<XBContextDialect>();
  assert(xbCtx && "null XBContext");
  registerXBGDeclConcepts(xbCtx->getContext());
  registerXBGTypeConcepts(xbCtx->getContext());
  registerXBGStmtConcepts(xbCtx->getContext());
  registerXBGExprConcepts(xbCtx->getContext());
  xblangContext = &xbCtx->getContext();
  addInterface<XGBContextInterface>();
  initializeType();
  initializeDecl();
  initializeStmt();
  initializeExpr();
}

#include "xblang/Lang/XBLang/XLG/XBGDialect.cpp.inc"
