//===- Concepts.h - Base XBLang language concepts  ---------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares base XBLang language concepts.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_XLG_CONCEPTS_H
#define XBLANG_XLG_CONCEPTS_H

#include "xblang/Interfaces/Concept.h"
#include "xblang/Interfaces/SymbolTable.h"

#include "xblang/XLG/IR/XLGTypes.h"

#define GET_CLASS_DECL
#include "xblang/XLG/XLGConcepts.h.inc"

namespace xblang {
namespace xlg {
#define GET_REGISTRATION_DECL
#include "xblang/XLG/XLGConcepts.h.inc"
} // namespace xlg
} // namespace xblang

#endif // XBLANG_XLG_CONCEPTS_H
