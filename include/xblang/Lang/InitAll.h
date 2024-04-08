//===- InitAll.h - Initializes all in-tree extensions ------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares helper functions to initialize all in-tree extensions.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_LANG_INITALL_H
#define XBLANG_LANG_INITALL_H

#include "xblang/Lang/Meta/Init.h"
#include "xblang/Lang/Parallel/Init.h"
#include "xblang/Lang/XBLang/Init.h"
#include "xblang/XLG/Init.h"

namespace xblang {
/// Registers all extensions in the given registry.
inline void registerXBLang(mlir::DialectRegistry &registry) {
  xlg::registerXLG(registry);
  xbg::registerXBG(registry);
  par::registerPar(registry);
  meta::registerMeta(registry);
}

/// Registers all extensions in the given context.
inline void registerXBLang(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  registerXBLang(registry);
  context.appendDialectRegistry(registry);
}
} // namespace xblang

#endif // XBLANG_LANG_INITALL_H
