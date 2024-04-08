//===- mlir-opt.cpp - MLIR Optimizer Driver -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for mlir-opt for when built as standalone binary.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "xblang/Basic/ContextDialect.h"
#include "xblang/Dialect/Parallel/IR/Dialect.h"
#include "xblang/Dialect/XBLang/IR/Dialect.h"
#include "xblang/Lang/Parallel/Frontend/Options.h"
#include "xblang/Lang/XBLang/XLG/XBGDialect.h"
#include "xblang/XLG/IR/XLGDialect.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;
using namespace mlir;

int main(int argc, char **argv) {
  ::xblang::par::registerParOptions();
  DialectRegistry registry;
  registry.insert<xblang::xb::XBLangDialect>();
  registry.insert<mlir::par::ParDialect>();
  registry.insert<::xblang::xlg::XLGDialect>();
  registry.insert<::xblang::XBContextDialect>();
  registry.insert<::xblang::xbg::XBGDialect>();
  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "XBLang MLIR modular optimizer driver\n", registry));
}
