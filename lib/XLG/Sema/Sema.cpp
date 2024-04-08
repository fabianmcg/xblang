//===- Sema.cpp - XLG semantic checker ---------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines XLG semantic checker.
//
//===----------------------------------------------------------------------===//

#include "xblang/XLG/Sema/Sema.h"
#include "mlir/IR/DialectRegistry.h"
#include "xblang/Basic/Context.h"
#include "xblang/Interfaces/Sema.h"
#include "xblang/Sema/Sema.h"
#include "xblang/XLG/IR/XLGDialect.h"
#include "xblang/XLG/Interfaces.h"

using namespace xblang;
using namespace xblang::sema;
using namespace xblang::xlg;

static void populateXLGSemaPatterns(GenericPatternSet &patterns);

namespace {
class SemaInterface : public xblang::SemaDialectInterface {
public:
  using xblang::SemaDialectInterface::SemaDialectInterface;

  void populateSemaPatterns(GenericPatternSet &patterns,
                            TypeSystem &typeSystem) const override {
    populateXLGSemaPatterns(patterns);
  }
};

struct RegionVerifier : public SemaOpPattern<RegionOp> {
  using Base::Base;
};

struct TemplateVerifier : public SemaOpPattern<TemplateOp> {
  using Base::Base;

  SemaResult check(Op op, Status status, xblang::SymbolTable *symTable,
                   SemaDriver &driver) const final;
};

struct TemplateInstanceVerifier : public SemaOpPattern<TemplateInstanceOp> {
  using Base::Base;

  SemaResult check(Op op, Status status, xblang::SymbolTable *symTable,
                   SemaDriver &driver) const final;
};
} // namespace

//===----------------------------------------------------------------------===//
// XLG patterns
//===----------------------------------------------------------------------===//

SemaResult TemplateVerifier::check(Op op, Status status,
                                   xblang::SymbolTable *symTable,
                                   SemaDriver &driver) const {
  op.setUsr(op.getSymId());
  driver.setUSR(op.getUSR(), op);
  return success();
}

SemaResult TemplateInstanceVerifier::check(Op op, Status status,
                                           xblang::SymbolTable *symTable,
                                           SemaDriver &driver) const {
  TemplateOp templateOp = dyn_cast_or_null<TemplateOp>(
      driver.lookupUSR(op.getUsrAttr().getRootReference()));
  if (!templateOp)
    return op.emitError("referenced template doesn't exist");
  auto &body = op.getBodyRegion();
  // Use a different builder to not alert the driver of new ops.
  OpBuilder builder(op.getContext());
  builder.cloneRegionBefore(templateOp.getRegion(), body, body.end());
  for (size_t i = 0; i < body.getNumArguments(); ++i) {
    driver.replaceAllUsesWith(body.getArgument(i), op.getArguments()[i]);
    body.eraseArgument(i);
  }
  for (auto symT : body.front().getOps<SymbolTableInterface>()) {
    if (failed(driver.buildTables(symT, symTable)))
      return op.emitError("failed creating the symbol table");
  }
  if (SemaResult result = driver.checkRegions(op, symTable);
      !result.succeeded())
    return result;
  return success();
}

//===----------------------------------------------------------------------===//
// XLG CG API
//===----------------------------------------------------------------------===//

void populateXLGSemaPatterns(GenericPatternSet &patterns) {
  patterns.add<RegionVerifier, TemplateVerifier, TemplateInstanceVerifier>(
      patterns.getContext()->getMLIRContext());
}

void xblang::xlg::registerXLGSemaInterface(mlir::DialectRegistry &registry) {
  registry.insert<XLGDialect>();
  registry.addExtension(+[](mlir::MLIRContext *ctx, XLGDialect *dialect) {
    dialect->addInterfaces<SemaInterface>();
  });
}

void xblang::xlg::registerXLGSemaInterface(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  registerXLGSemaInterface(registry);
  context.appendDialectRegistry(registry);
}
