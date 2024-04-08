//===- Passes.h - Semantic checker passes -----------------------*- C++ -*-===//
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

#ifndef XBLANG_SEMA_PASSES_H
#define XBLANG_SEMA_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xblang {
namespace sema {
#define GEN_PASS_DECL
#include "xblang/Sema/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "xblang/Sema/Passes.h.inc"
} // namespace sema
} // namespace xblang

#endif // XBLANG_SEMA_PASSES_H
