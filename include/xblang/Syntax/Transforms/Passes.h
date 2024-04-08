//===- Passes.h - Syntax passes  --------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_SYNTAX_TRANSFORMS_PASSES_H
#define XBLANG_SYNTAX_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xblang {
namespace syntaxgen {
#define GEN_PASS_DECL
#include "xblang/Syntax/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "xblang/Syntax/Transforms/Passes.h.inc"
} // namespace syntaxgen
} // namespace xblang

#endif // XBLANG_SYNTAX_TRANSFORMS_PASSES_H
