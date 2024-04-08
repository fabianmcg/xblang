//===- Init.h - Init syntax --------------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares functions for registering the syntax of an extension.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_LANG_XBLANG_SYNTAX_INIT_H
#define XBLANG_LANG_XBLANG_SYNTAX_INIT_H

namespace mlir {
class DialectRegistry;
class MLIRContext;
} // namespace mlir

namespace xblang {
namespace xbg {
/// Registers the XBG syntax dialect extension in the given registry.
void registerXBGSyntaxInterface(mlir::DialectRegistry &registry);
/// Registers the XBG syntax dialect extension in the given context.
void registerXBGSyntaxInterface(mlir::MLIRContext &context);
} // namespace xbg
} // namespace xblang

#endif // XBLANG_LANG_XBLANG_SYNTAX_INIT_H
