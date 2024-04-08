//===- LexTypes.h - Lex dialect types  ---------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Lex dialect Types.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_SYNTAX_IR_SYNTAXTYPES_H
#define XBLANG_SYNTAX_IR_SYNTAXTYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"

#define GET_TYPEDEF_CLASSES
#include "xblang/Syntax/IR/SyntaxOpsTypes.h.inc"

#endif // XBLANG_SYNTAX_IR_SYNTAXTYPES_H
