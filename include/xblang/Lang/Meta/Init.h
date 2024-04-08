//===- Init.h - Initializes all the Meta components --------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares initialization functions for par.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_LANG_META_INIT_H
#define XBLANG_LANG_META_INIT_H

#include "mlir/IR/DialectRegistry.h"
#include "xblang/Lang/Meta/Syntax/Init.h"

namespace xblang {
namespace meta {
/// Registers the meta extension in the given registry.
inline void registerMeta(mlir::DialectRegistry &registry) {
  registerMetaSyntaxInterface(registry);
}

/// Registers the meta extension in the given context.
inline void registerMeta(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  registerMeta(registry);
  context.appendDialectRegistry(registry);
}
} // namespace meta
} // namespace xblang

#endif // XBLANG_LANG_META_INIT_H
