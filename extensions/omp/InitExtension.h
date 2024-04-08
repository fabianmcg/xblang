//===- InitExtension.h - Init the extension ----------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares functions for registering the extension.
//
//===----------------------------------------------------------------------===//

#ifndef OMP_INITEXTENSION_H
#define OMP_INITEXTENSION_H

#include "mlir/IR/DialectRegistry.h"

namespace mlir {
class DialectRegistry;
class MLIRContext;
} // namespace mlir

namespace omp {
/// Registers the omp syntax dialect extension in the given registry.
void registerSyntaxInterface(mlir::DialectRegistry &registry);
/// Registers the omp syntax dialect extension in the given context.
void registerSyntaxInterface(mlir::MLIRContext &context);

/// Registers the omp sema dialect extension in the given registry.
void registerSemaInterface(mlir::DialectRegistry &registry);
/// Registers the omp sema dialect extension in the given context.
void registerSemaInterface(mlir::MLIRContext &context);

/// Registers the omp CG dialect extension in the given registry.
void registerCodeGenInterface(mlir::DialectRegistry &registry);
/// Registers the omp CG dialect extension in the given context.
void registerCodeGenInterface(mlir::MLIRContext &context);

/// Registers the omp extension in the given registry.
inline void registerOMPExtension(mlir::DialectRegistry &registry) {
  registerSyntaxInterface(registry);
  registerSemaInterface(registry);
  registerCodeGenInterface(registry);
}

/// Registers the omp extension in the given context.
inline void registerOMPExtension(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  registerOMPExtension(registry);
  context.appendDialectRegistry(registry);
}
} // namespace omp

#endif // OMP_INITEXTENSION_H
