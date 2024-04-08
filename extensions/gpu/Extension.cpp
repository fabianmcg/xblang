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

#include "gpu/GPUExtension.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "xblang/Codegen/Codegen.h"
#include "xblang/Interfaces/Codegen.h"
#include "xblang/Interfaces/Sema.h"
#include "xblang/Sema/Sema.h"
#include "xblang/XLG/Concepts.h"
#include "xblang/XLG/IR/XLGDialect.h"
#include "llvm/Support/Error.h"

using namespace ::gpu;
using namespace mlir;

namespace {
//===----------------------------------------------------------------------===//
// GPULaunchChecker
//===----------------------------------------------------------------------===//
class GPULaunchChecker
    : public xblang::sema::SemaOpPattern<mlir::gpu::LaunchOp> {
public:
  using Base::Base;
};

//===----------------------------------------------------------------------===//
// GPULaunchCG
//===----------------------------------------------------------------------===//
class GPULaunchCG : public xblang::codegen::OpCGPattern<mlir::gpu::LaunchOp> {
public:
  using Base::Base;
  xblang::codegen::CGResult
  generate(Op op, xblang::codegen::CGDriver &driver) const override;
};

//===----------------------------------------------------------------------===//
// GPUSema
//===----------------------------------------------------------------------===//
class GPUSema : public xblang::SemaDialectInterface {
public:
  using xblang::SemaDialectInterface::SemaDialectInterface;

  void populateSemaPatterns(xblang::GenericPatternSet &set,
                            xblang::TypeSystem &typeSystem) const override {
    set.add<GPULaunchChecker>(set.getMLIRContext(), 5);
  }
};

//===----------------------------------------------------------------------===//
// GPUCodeGen
//===----------------------------------------------------------------------===//
class GPUCodeGen : public xblang::CodegenDialectInterface {
public:
  using xblang::CodegenDialectInterface::CodegenDialectInterface;

  mlir::LogicalResult
  populateCodegenPatterns(xblang::GenericPatternSet &patterns,
                          mlir::TypeConverter *converter) const override {
    patterns.add<GPULaunchCG>(patterns.getMLIRContext(), converter);
    return success();
  }
};
} // namespace

xblang::codegen::CGResult
GPULaunchCG::generate(Op op, xblang::codegen::CGDriver &driver) const {
  for (auto &block : op.getRegion().getBlocks())
    for (auto &op : llvm::make_early_inc_range(block.getOperations()))
      driver.genOp(&op);
  return op.getOperation();
}

void ::gpu::registerSemaInterface(mlir::DialectRegistry &registry) {
  registry.insert<mlir::gpu::GPUDialect>();
  registry.addExtension(
      +[](mlir::MLIRContext *ctx, mlir::gpu::GPUDialect *dialect) {
        dialect->addInterfaces<GPUSema>();
      });
}

void ::gpu::registerSemaInterface(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  registerSemaInterface(registry);
  context.appendDialectRegistry(registry);
}

void ::gpu::registerCodeGenInterface(mlir::DialectRegistry &registry) {
  registry.insert<mlir::gpu::GPUDialect>();
  registry.addExtension(
      +[](mlir::MLIRContext *ctx, mlir::gpu::GPUDialect *dialect) {
        dialect->addInterfaces<GPUCodeGen>();
      });
}

void ::gpu::registerCodeGenInterface(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  registerCodeGenInterface(registry);
  context.appendDialectRegistry(registry);
}
