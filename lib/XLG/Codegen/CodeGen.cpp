//===- CodeGen.cpp - XLG code generator --------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines XLG code generation.
//
//===----------------------------------------------------------------------===//

#include "xblang/XLG/Codegen/Codegen.h"
#include "mlir/IR/DialectRegistry.h"
#include "xblang/Basic/Context.h"
#include "xblang/Codegen/Codegen.h"
#include "xblang/Interfaces/Codegen.h"
#include "xblang/XLG/IR/XLGDialect.h"
#include "xblang/XLG/Interfaces.h"

using namespace xblang;
using namespace xblang::codegen;
using namespace xblang::xlg;

namespace {
class CGInterface : public xblang::CodegenDialectInterface {
public:
  using xblang::CodegenDialectInterface::CodegenDialectInterface;

  LogicalResult
  populateCodegenPatterns(GenericPatternSet &patterns,
                          TypeConverter *converter) const override {
    populateCGPatterns(patterns);
    return success();
  }
};

struct RegionOpOpCG : public OpCGPattern<RegionOp> {
  using Base::Base;

  CGResult generate(Op op, CGDriver &driver) const final;
};

struct RegionReturnOpCG : public OpCGPattern<RegionReturnOp> {
  using Base::Base;

  CGResult generate(Op op, CGDriver &driver) const final;
};

struct ReturnOpCG : public OpCGPattern<ReturnOp> {
  using Base::Base;

  CGResult generate(Op op, CGDriver &driver) const final;
};

struct TemplateCG : public OpCGPattern<TemplateOp> {
  using Base::Base;

  CGResult generate(Op op, CGDriver &driver) const final;
};

struct TypeCG : public InterfaceCGPattern<TypeAttrInterface> {
  using Base::Base;

  CGResult generate(Interface op, CGDriver &driver) const final;
};

} // namespace

//===----------------------------------------------------------------------===//
// XLG patterns
//===----------------------------------------------------------------------===//

CGResult RegionOpOpCG::generate(Op op, CGDriver &driver) const {
  auto ret = cast<RegionReturnOp>(op.getBody(0)->getTerminator());
  driver.inlineBlockBefore(op.getBody(), op->getBlock(),
                           op.getOperation()->getIterator());
  ValueRange expr = ret.getExpr();
  Value val{};
  if (expr.size() > 0) {
    driver.replaceOp(op, expr);
    val = expr[0];
  } else
    driver.eraseOp(op);
  return val;
}

CGResult RegionReturnOpCG::generate(Op op, CGDriver &driver) const {
  driver.eraseOp(op);
  return nullptr;
}

CGResult ReturnOpCG::generate(Op op, CGDriver &driver) const {
  driver.eraseOp(op);
  return nullptr;
}

CGResult TemplateCG::generate(Op op, CGDriver &driver) const {
  driver.eraseOp(op);
  return nullptr;
}

CGResult TypeCG::generate(Interface op, CGDriver &driver) const {
  auto type = op.getType();
  driver.eraseOp(op);
  return type;
}

//===----------------------------------------------------------------------===//
// XLG CG API
//===----------------------------------------------------------------------===//

void xblang::xlg::populateCGPatterns(GenericPatternSet &patterns) {
  patterns.add<RegionOpOpCG, RegionReturnOpCG, ReturnOpCG, TemplateCG, TypeCG>(
      patterns.getContext()->getMLIRContext());
}

void xblang::xlg::registerXLGCGInterface(mlir::DialectRegistry &registry) {
  registry.insert<XLGDialect>();
  registry.addExtension(+[](mlir::MLIRContext *ctx, XLGDialect *dialect) {
    dialect->addInterfaces<CGInterface>();
  });
}

void xblang::xlg::registerXLGCGInterface(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  registerXLGCGInterface(registry);
  context.appendDialectRegistry(registry);
}
