//===- XBGExpr.h - XBG Expr constructs --------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the XBG Expr constructs and concepts.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_LANG_XBLANG_XLG_XBGEXPR_H
#define XBLANG_LANG_XBLANG_XLG_XBGEXPR_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "xblang/XLG/Concepts.h"
#include "xblang/XLG/IR/XLGTraits.h"
#include "xblang/XLG/IR/XLGTypes.h"

#include "xblang/Dialect/XBLang/IR/Enums.h"

namespace xblang {
class XBContext;
}

#define GET_CLASS_DECL
#include "xblang/Lang/XBLang/XLG/XBGConceptExpr.h.inc"

#define GET_OP_CLASSES
#include "xblang/Lang/XBLang/XLG/XBGExpr.h.inc"

#endif // XBLANG_LANG_XBLANG_XLG_XBGEXPR_H
