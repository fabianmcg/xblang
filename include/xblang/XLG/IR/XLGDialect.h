//===- XLGDialect.h - XLG dialect -------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the XLG dialect.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_XLG_IR_XLGDIALECT_H
#define XBLANG_XLG_IR_XLGDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "xblang/Interfaces/SymbolTable.h"
#include "xblang/XLG/IR/XLGTypes.h"
#include "xblang/XLG/Interfaces.h"

namespace mlir {
class PatternRewriter;
}

namespace xblang {
namespace xlg {
/// Class for parsing template parameters.
struct TemplateParam {
  mlir::LocationAttr loc;
  llvm::StringRef identifier;
  ConceptType conceptClass;
  mlir::Value init;
};
} // namespace xlg
} // namespace xblang

#include "xblang/XLG/IR/XLGOpsDialect.h.inc"

#define GET_ATTRDEF_CLASSES
#include "xblang/XLG/IR/XLGOpsAttributes.h.inc"

#define GET_OP_CLASSES
#include "xblang/XLG/IR/XLGOps.h.inc"

#endif // XBLANG_XLG_IR_XLGDIALECT_H
