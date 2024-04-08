//===- Eval.cpp - XBG op evaluator --------------------------------*-C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares function for registering op evaluation patterns.
//
//===----------------------------------------------------------------------===//

#include "xblang/Lang/XBLang/Eval/Eval.h"
#include "mlir/IR/DialectRegistry.h"
#include "xblang/Basic/Pattern.h"
#include "xblang/Interfaces/Eval.h"
#include "xblang/Lang/XBLang/XLG/XBGDialect.h"
#include "xblang/Support/LLVM.h"

using namespace xblang;
using namespace xblang::xbg;

namespace {
class XGBEvalInterface : public xblang::EvalDialectInterface {
public:
  using xblang::EvalDialectInterface::EvalDialectInterface;

  void populateEvalPatterns(GenericPatternSet &patterns) const override {
    populateExprEvalPatterns(patterns);
  }
};
} // namespace

void xblang::xbg::registerXBGEvalInterface(mlir::DialectRegistry &registry) {
  registry.insert<XBGDialect>();
  registry.addExtension(+[](mlir::MLIRContext *ctx, XBGDialect *dialect) {
    dialect->addInterfaces<XGBEvalInterface>();
  });
}

void xblang::xbg::registerXBGEvalInterface(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  registerXBGEvalInterface(registry);
  context.appendDialectRegistry(registry);
}
