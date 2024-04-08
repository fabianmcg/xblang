//===- Extension.cpp - Defines the extension ---------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the extension.
//
//===----------------------------------------------------------------------===//

#include "omp/Extension.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "xblang/Codegen/Codegen.h"
#include "xblang/Interfaces/Codegen.h"
#include "xblang/Interfaces/Sema.h"
#include "xblang/Sema/Sema.h"
#include "xblang/XLG/Concepts.h"
#include "xblang/XLG/IR/XLGDialect.h"
#include "llvm/Support/Error.h"

using namespace ::omp;
using namespace mlir;

namespace {
//===----------------------------------------------------------------------===//
// OMPParallelChecker
//===----------------------------------------------------------------------===//
class OMPParallelChecker
    : public xblang::sema::SemaOpPattern<mlir::omp::ParallelOp> {
public:
  using Base::Base;
};

//===----------------------------------------------------------------------===//
// OMPParallelCG
//===----------------------------------------------------------------------===//
class OMPParallelCG
    : public xblang::codegen::OpCGPattern<mlir::omp::ParallelOp> {
public:
  using Base::Base;
  xblang::codegen::CGResult
  generate(Op op, xblang::codegen::CGDriver &driver) const override;
};

//===----------------------------------------------------------------------===//
// OMPSema
//===----------------------------------------------------------------------===//
class OMPSema : public xblang::SemaDialectInterface {
public:
  using xblang::SemaDialectInterface::SemaDialectInterface;

  void populateSemaPatterns(xblang::GenericPatternSet &set,
                            xblang::TypeSystem &typeSystem) const override {
    set.add<OMPParallelChecker>(set.getMLIRContext(), 5);
  }
};

//===----------------------------------------------------------------------===//
// OMPCodeGen
//===----------------------------------------------------------------------===//
class OMPCodeGen : public xblang::CodegenDialectInterface {
public:
  using xblang::CodegenDialectInterface::CodegenDialectInterface;

  mlir::LogicalResult
  populateCodegenPatterns(xblang::GenericPatternSet &patterns,
                          mlir::TypeConverter *converter) const override {
    patterns.add<OMPParallelCG>(patterns.getMLIRContext(), converter);
    return success();
  }
};
} // namespace

xblang::codegen::CGResult
OMPParallelCG::generate(Op op, xblang::codegen::CGDriver &driver) const {
  for (auto &block : op.getRegion().getBlocks())
    for (auto &op : llvm::make_early_inc_range(block.getOperations()))
      driver.genOp(&op);
  return op.getOperation();
}

void ::omp::registerSemaInterface(mlir::DialectRegistry &registry) {
  registry.insert<mlir::omp::OpenMPDialect>();
  registry.addExtension(
      +[](mlir::MLIRContext *ctx, mlir::omp::OpenMPDialect *dialect) {
        dialect->addInterfaces<OMPSema>();
      });
}

void ::omp::registerSemaInterface(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  registerSemaInterface(registry);
  context.appendDialectRegistry(registry);
}

void ::omp::registerCodeGenInterface(mlir::DialectRegistry &registry) {
  registry.insert<mlir::omp::OpenMPDialect>();
  registry.addExtension(
      +[](mlir::MLIRContext *ctx, mlir::omp::OpenMPDialect *dialect) {
        dialect->addInterfaces<OMPCodeGen>();
      });
}

void ::omp::registerCodeGenInterface(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  registerCodeGenInterface(registry);
  context.appendDialectRegistry(registry);
}
