//===- SyntaxDialect.h - Syntax dialect -------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Syntax dialect.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_SYNTAX_IR_SYNTAXDIALECT_H
#define XBLANG_SYNTAX_IR_SYNTAXDIALECT_H

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

#include "xblang/Syntax/IR/CharClass.h"
#include "xblang/Syntax/IR/SyntaxTypes.h"

namespace mlir {
class PatternRewriter;
}

#include "xblang/Syntax/IR/SyntaxOpsEnums.h.inc"

#include "xblang/Syntax/IR/SyntaxOpsDialect.h.inc"

#define GET_ATTRDEF_CLASSES
#include "xblang/Syntax/IR/SyntaxOpsAttributes.h.inc"

#define GET_OP_CLASSES
#include "xblang/Syntax/IR/SyntaxOps.h.inc"

#endif // XBLANG_SYNTAX_IR_SYNTAXDIALECT_H
