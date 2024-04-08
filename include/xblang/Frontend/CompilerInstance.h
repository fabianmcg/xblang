//===- CompilerInstance.cpp - XBLang compiler instance -----------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the XBLang compiler instance.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_FRONTEND_COMPILERINSTANCE_H
#define XBLANG_FRONTEND_COMPILERINSTANCE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "xblang/Basic/SourceManager.h"
#include "xblang/Syntax/SyntaxContext.h"
#include <memory>
#include <vector>

namespace llvm {
class SourceMgr;
class StringRef;
} // namespace llvm

namespace mlir {
class MLIRContext;
class Block;
} // namespace mlir

namespace xblang {
class XBContext;
class CompilerInvocation;

namespace xb {
class XBLangTypeSystem;
}

/// XBLang compiler instance
class CompilerInstance {
public:
  CompilerInstance(CompilerInvocation &invocation);
  ~CompilerInstance();

  /// Returns the compiler invocation.
  const CompilerInvocation &getInvocation() const { return invocation; }

  /// Returns whether the MLIR context is present.
  bool hasMLIRContext() { return mlirContext.get(); }

  /// Returns the MLIR context.
  mlir::MLIRContext &getMLIRContext() { return *mlirContext; }

  /// Returns the source manager.
  SourceManager &getSourceManager() { return *sourceManager; }

  /// Returns the MLIR module being used by this instance.
  mlir::ModuleOp getModule() { return *module; }

  /// Returns the XBLang context.
  XBContext *getXBLangContext() { return xblangContext; }

  xblang::xb::XBLangTypeSystem &getTypeContext() { return *typeContext; }

  static std::unique_ptr<llvm::raw_ostream> getOutput(llvm::Twine filePath);

  /// Runs the instance.
  int run();

private:
  /// Runs the parsing stage.
  int parse();
  CompilerInvocation &invocation;
  std::unique_ptr<mlir::MLIRContext> mlirContext;
  std::unique_ptr<SourceManager> sourceManager;
  std::unique_ptr<xblang::xb::XBLangTypeSystem> typeContext;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  XBContext *xblangContext{};
};
} // namespace xblang
#endif
