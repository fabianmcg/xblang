//===- Init.h - Initializes all the XBG components ---------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares initialization functions for XBG.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_LANG_XBLANG_INIT_H
#define XBLANG_LANG_XBLANG_INIT_H

#include "mlir/IR/DialectRegistry.h"
#include "xblang/Lang/XBLang/Codegen/Codegen.h"
#include "xblang/Lang/XBLang/Eval/Eval.h"
#include "xblang/Lang/XBLang/Sema/Sema.h"
#include "xblang/Lang/XBLang/Syntax/Init.h"
#include "xblang/Lang/XBLang/XLG/XBGDialect.h"

namespace xblang {
namespace xbg {
/// Registers XBG in the given registry.
inline void registerXBG(mlir::DialectRegistry &registry) {
  registry.insert<XBGDialect>();
  registerXBGSemaInterface(registry);
  registerXBGCGInterface(registry);
  registerXBGSyntaxInterface(registry);
  registerXBGEvalInterface(registry);
}

/// Registers XBG in the given context.
inline void registerXBG(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  registerXBG(registry);
  context.appendDialectRegistry(registry);
}
} // namespace xbg
} // namespace xblang

#endif // XBLANG_LANG_XBLANG_INIT_H
