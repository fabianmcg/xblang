//===- XBGType.cpp - XBG type constructs ------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements XBG Type constructs.
//
//===----------------------------------------------------------------------===//

#include "xblang/Lang/XBLang/XLG/XBGType.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "xblang/Basic/Context.h"
#include "xblang/Lang/XBLang/XLG/XBGDialect.h"
#include "xblang/XLG/Concepts.h"
#include "xblang/XLG/Interfaces.h"

using namespace mlir;
using namespace xblang::xbg;

//===----------------------------------------------------------------------===//
// XBGType
//===----------------------------------------------------------------------===//

void XBGDialect::initializeType() {
  addOperations<
#define GET_OP_LIST
#include "xblang/Lang/XBLang/XLG/XBGType.cpp.inc"
      >();
  xblangContext->registerConstructs<
#define GET_OP_LIST
#include "xblang/Lang/XBLang/XLG/XBGType.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "xblang/Lang/XBLang/XLG/XBGType.cpp.inc"

#define GET_CLASS_DEF
#include "xblang/Lang/XBLang/XLG/XBGConceptType.cpp.inc"

namespace xblang {
namespace xbg {
#define GET_REGISTRATION_DEF
#include "xblang/Lang/XBLang/XLG/XBGConceptType.cpp.inc"
} // namespace xbg
} // namespace xblang

void BuiltinType::build(::mlir::OpBuilder &odsBuilder,
                        ::mlir::OperationState &odsState,
                        ::mlir::Type conceptClass, ::mlir::Type type) {
  build(odsBuilder, odsState, conceptClass, mlir::TypeAttr::get(type));
}
