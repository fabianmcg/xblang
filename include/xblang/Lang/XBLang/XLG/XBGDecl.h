//===- XBGDecl.h - XBG Decl constructs --------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the XBG Decl constructs and concepts.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_LANG_XBLANG_XLG_XBGDECL_H
#define XBLANG_LANG_XBLANG_XLG_XBGDECL_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "xblang/XLG/Concepts.h"
#include "xblang/XLG/IR/XLGTraits.h"
#include "xblang/XLG/IR/XLGTypes.h"

namespace xblang {
class XBContext;
}

#define GET_CLASS_DECL
#include "xblang/Lang/XBLang/XLG/XBGConceptDecl.h.inc"

#define GET_OP_CLASSES
#include "xblang/Lang/XBLang/XLG/XBGDecl.h.inc"

#endif // XBLANG_LANG_XBLANG_XLG_XBGDECL_H
