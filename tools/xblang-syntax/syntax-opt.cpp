//===- syntax-opt.cpp - Syntax Optimizer Driver ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for syntax-opt.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "xblang/Syntax/IR/SyntaxDialect.h"
#include "xblang/Syntax/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;
using namespace mlir;

int main(int argc, char **argv) {
  registerTransformsPasses();
  DialectRegistry registry;
  registry.insert<xblang::syntaxgen::SyntaxDialect>();
  xblang::syntaxgen::registerSyntaxPasses();
  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "Syntax Gen MLIR modular optimizer driver\n", registry));
}
