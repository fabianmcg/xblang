//===- Codegen.h - Code generation interfaces --------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares code generation interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_INTREFACES_CODEGEN_H
#define XBLANG_INTREFACES_CODEGEN_H

#include "mlir/IR/DialectInterface.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
class TypeConverter;
}

namespace xblang {
class GenericPatternSet;

/// A dialect interface for participating in code generation.
class CodegenDialectInterface
    : public mlir::DialectInterface::Base<CodegenDialectInterface> {
public:
  CodegenDialectInterface(mlir::Dialect *dialect) : Base(dialect) {}

  /// Adds patterns to the pattern set.
  virtual mlir::LogicalResult
  populateCodegenPatterns(GenericPatternSet &patterns,
                          mlir::TypeConverter *converter) const = 0;
};
} // namespace xblang

#endif // XBLANG_INTREFACES_CODEGEN_H
