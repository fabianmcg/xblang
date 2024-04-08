//===- Builder.h - XLG builder  ----------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the XLG builder.
//
//===----------------------------------------------------------------------===//

#include "xblang/XLG/Builder.h"
#include "xblang/Interfaces/SymbolTable.h"
#include "xblang/XLG/Interfaces.h"

using namespace xblang::xlg;

TemplateOp XLGBuilder::createTemplate(mlir::Location loc, ConceptType kind,
                                      mlir::Block *block,
                                      ArrayRef<TemplateParam> parameters,
                                      Op ret) {
  assert(ret && "invalid parameter");
  Op symOp = ret;
  auto sym = dyn_cast_or_null<::xblang::Symbol>(symOp);
  if (!sym) {
    ret->emitError("op didn't implement the SymbolInterface");
    return {};
  }
  SmallVector<mlir::Type> ins;
  for (auto &param : parameters)
    ins.push_back(param.conceptClass);
  TemplateOp tmpl =
      create<TemplateOp>(loc, ("_$" + sym.getIdentifier().getValue()).str(),
                         getFunctionType(ins, {kind}), block);
  auto grd = guard(*this, block);
  create<ReturnOp>(loc, ret->getResult(0));
  return tmpl;
}
