//===- Passes.h - Code generation passes ------------------------*- C++ -*-===//
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

#ifndef XBLANG_CODEGEN_PASSES_H
#define XBLANG_CODEGEN_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xblang {
namespace codegen {
#define GEN_PASS_DECL
#include "xblang/Codegen/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "xblang/Codegen/Passes.h.inc"
} // namespace codegen
} // namespace xblang

#endif // XBLANG_CODEGEN_PASSES_H
