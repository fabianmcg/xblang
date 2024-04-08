//===- Concepts.cpp - Base XBLang language concepts  -------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines base XBLang language concepts.
//
//===----------------------------------------------------------------------===//

#include "xblang/XLG/Concepts.h"
#include "xblang/Basic/Context.h"
#include "xblang/XLG/IR/XLGTypes.h"

using namespace xblang;
using namespace xblang::xlg;

#define GET_CLASS_DEF
#include "xblang/XLG/XLGConcepts.cpp.inc"

namespace xblang {
namespace xlg {
#define GET_REGISTRATION_DEF
#include "xblang/XLG/XLGConcepts.cpp.inc"
} // namespace xlg
} // namespace xblang
