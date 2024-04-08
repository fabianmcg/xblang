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

#ifndef GPU_INITEXTENSION_H
#define GPU_INITEXTENSION_H

#include "mlir/IR/DialectRegistry.h"

namespace mlir {
class DialectRegistry;
class MLIRContext;
} // namespace mlir

namespace gpu {
/// Registers the gpu syntax dialect extension in the given registry.
void registerSyntaxInterface(mlir::DialectRegistry &registry);
/// Registers the gpu syntax dialect extension in the given context.
void registerSyntaxInterface(mlir::MLIRContext &context);

/// Registers the omp sema dialect extension in the given registry.
void registerSemaInterface(mlir::DialectRegistry &registry);
/// Registers the omp sema dialect extension in the given context.
void registerSemaInterface(mlir::MLIRContext &context);

/// Registers the omp CG dialect extension in the given registry.
void registerCodeGenInterface(mlir::DialectRegistry &registry);
/// Registers the omp CG dialect extension in the given context.
void registerCodeGenInterface(mlir::MLIRContext &context);

/// Registers the gpu extension in the given registry.
inline void registerGPUExtension(mlir::DialectRegistry &registry) {
  registerSyntaxInterface(registry);
  registerSemaInterface(registry);
  registerCodeGenInterface(registry);
}

/// Registers the gpu extension in the given context.
inline void registerGPUExtension(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  registerGPUExtension(registry);
  context.appendDialectRegistry(registry);
}
} // namespace gpu

#endif // GPU_INITEXTENSION_H
