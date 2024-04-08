//===- Type.cpp - XBG code gen patterns for type constructs -----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements XBG code generation patterns for type constructs.
//
//===----------------------------------------------------------------------===//

#include "xblang/Lang/XBLang/Codegen/Codegen.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "xblang/Basic/Context.h"
#include "xblang/Codegen/Codegen.h"
#include "xblang/Lang/XBLang/XLG/XBGDialect.h"
#include "xblang/Lang/XBLang/XLG/XBGType.h"
#include "xblang/XLG/Concepts.h"
#include "xblang/XLG/Interfaces.h"

using namespace xblang;
using namespace xblang::codegen;
using namespace xblang::xbg;

//===----------------------------------------------------------------------===//
// XBG code generation patterns
//===----------------------------------------------------------------------===//

void xblang::xbg::populateTypeCGPatterns(GenericPatternSet &patterns,
                                         const mlir::TypeConverter *converter) {
}
