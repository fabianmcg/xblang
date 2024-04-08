//===- XBGDialect.h - XBG dialect -------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the XBG dialect.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_LANG_XBLANG_XLG_XBGDIALECT_H
#define XBLANG_LANG_XBLANG_XLG_XBGDIALECT_H

#include "mlir/IR/Dialect.h"

namespace xblang {
class XBContext;
}

#include "xblang/Lang/XBLang/XLG/XBGDialect.h.inc"

namespace xblang {
namespace xbg {
#define GET_REGISTRATION_DECL
#include "xblang/Lang/XBLang/XLG/XBGConceptDecl.h.inc"
#define GET_REGISTRATION_DECL
#include "xblang/Lang/XBLang/XLG/XBGConceptType.h.inc"
#define GET_REGISTRATION_DECL
#include "xblang/Lang/XBLang/XLG/XBGConceptStmt.h.inc"
#define GET_REGISTRATION_DECL
#include "xblang/Lang/XBLang/XLG/XBGConceptExpr.h.inc"
} // namespace xbg
} // namespace xblang

#endif // XBLANG_LANG_XBLANG_XLG_XBGDIALECT_H
