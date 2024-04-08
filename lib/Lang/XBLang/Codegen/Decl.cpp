//===- Decl.cpp - XBG code gen patterns for decl constructs -----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements XBG code generation patterns for decl constructs.
//
//===----------------------------------------------------------------------===//

#include "xblang/Lang/XBLang/Codegen/Codegen.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "xblang/Basic/Context.h"
#include "xblang/Codegen/Codegen.h"
#include "xblang/Lang/XBLang/XLG/XBGDecl.h"
#include "xblang/Lang/XBLang/XLG/XBGDialect.h"
#include "xblang/Lang/XBLang/XLG/XBGType.h"
#include "xblang/XLG/Concepts.h"
#include "xblang/XLG/IR/XLGDialect.h"
#include "xblang/XLG/Interfaces.h"

#include "xblang/Dialect/XBLang/IR/XBLang.h"

using namespace xblang;
using namespace xblang::codegen;
using namespace xblang::xbg;

namespace {
//===----------------------------------------------------------------------===//
// FuncDeclCG
//===----------------------------------------------------------------------===//
struct FuncDeclCG : public OpCGPattern<FuncDecl> {
  using Base::Base;

  CGResult generate(Op op, CGDriver &driver) const final;
};

//===----------------------------------------------------------------------===//
// FuncDefCG
//===----------------------------------------------------------------------===//
struct FuncDefCG : public OpCGPattern<FuncDef> {
  using Base::Base;

  CGResult generate(Op op, CGDriver &driver) const final;
};

//===----------------------------------------------------------------------===//
// VarDeclCG
//===----------------------------------------------------------------------===//
struct VarDeclCG : public OpCGPattern<VarDecl> {
  using Base::Base;

  CGResult generate(Op op, CGDriver &driver) const final;
};
} // namespace

//===----------------------------------------------------------------------===//
// FuncDeclCG
//===----------------------------------------------------------------------===//

CGResult FuncDeclCG::generate(Op op, CGDriver &driver) const {
  auto fn = driver.create<xblang::xb::FunctionOp>(
      op.getLoc(), op.getUSR(),
      cast<FunctionType>(convertType(op.getTypeAttr().getValue())));
  if (op.getBodyRegion().empty()) {
    fn.eraseBody();
    fn.setVisibility(mlir::SymbolTable::Visibility::Private);
  } else {
    auto fnDef =
        cast<FuncDef>(cast<xlg::ReturnOp>(op.getBody()->getTerminator())
                          .getExpr()
                          .getDefiningOp());
    if (fnDef.getBodyRegion().empty()) {
      fn.getBody().front().erase();
      fn.setVisibility(mlir::SymbolTable::Visibility::Private);
    } else {
      for (auto [i, arg] : llvm::enumerate(fnDef.getArguments()))
        driver.mapValue(arg.getDefiningOp(), fn.getArgument(i));
      driver.inlineBlockBefore(op.getBody(), &fn.getBlocks().front(),
                               fn.getBlocks().front().end());
    }
  }
  driver.eraseOp(op);
  return fn.getOperation();
}

//===----------------------------------------------------------------------===//
// FuncDefCG
//===----------------------------------------------------------------------===//

CGResult FuncDefCG::generate(Op op, CGDriver &driver) const {
  if (!op.getBodyRegion().empty())
    driver.inlineBlockBefore(op.getBody(), op->getBlock(),
                             op->getBlock()->end());
  driver.eraseOp(op);
  return nullptr;
}

//===----------------------------------------------------------------------===//
// VarDeclCG
//===----------------------------------------------------------------------===//

CGResult VarDeclCG::generate(Op op, CGDriver &driver) const {
  auto type = driver.genValue(op.getValueType()).dyn_cast<mlir::Type>();
  assert(type && "invalid type");
  type = convertType(type);
  xb::VarKind kind = xb::VarKind::local;
  if (auto cep = op.getConceptClass().getType().getConceptClass();
      cep && isa<xbg::ParamDecl>(cep.getConcept()))
    kind = xb::VarKind::param;
  Value init = nullptr;
  if (auto expr = op.getExpr(); expr && kind == xb::VarKind::local)
    init = driver.genValue(expr).dyn_cast<Value>();
  if (kind == xb::VarKind::param) {
    Value paramDecl = driver.lookupValue(op.getOperation());
    assert(paramDecl && "invalid parameter");
    init = paramDecl;
  }
  return driver
      .replaceOpWithNewOp<xb::VarOp>(op, xb::ReferenceType::get(type),
                                     op.getSymId(), type, kind, init)
      .getResult();
}

//===----------------------------------------------------------------------===//
// XBG code generation patterns
//===----------------------------------------------------------------------===//

void xblang::xbg::populateDeclCGPatterns(GenericPatternSet &patterns,
                                         const mlir::TypeConverter *converter) {
  patterns.add<FuncDeclCG, FuncDefCG, VarDeclCG>(patterns.getMLIRContext(),
                                                 converter, 10);
}
