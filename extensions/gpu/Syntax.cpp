//===- Syntax.cpp - Defines the extension syntax -----------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the extension syntax.
//
//===----------------------------------------------------------------------===//

#include "xblang/Interfaces/Syntax.h"
#include "gpu/GPUExtension.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "xblang/Lang/XBLang/Syntax/Syntax.h"
#include "xblang/Syntax/ParsingCombinators.h"
#include "xblang/Syntax/SyntaxContext.h"
#include "xblang/XLG/Concepts.h"
#include "xblang/XLG/IR/XLGDialect.h"
#include "llvm/Support/Error.h"

using namespace ::gpu;

static void registerRegionCombinators(GPUParser &parser);

#include "gpu/GPUParser.cpp.inc"

using namespace xblang::syntax;
using namespace mlir;

namespace {
//===----------------------------------------------------------------------===//
// GPUSyntax
//===----------------------------------------------------------------------===//
class GPUSyntax : public xblang::SyntaxDialectInterface {
public:
  using xblang::SyntaxDialectInterface::SyntaxDialectInterface;

  void populateSyntax(xblang::XBContext *context,
                      xblang::SyntaxContext &syntaxContext) const override {
    llvm::StringRef kw = "gpu";
    syntaxContext.getOrRegisterLexer<xblang::XBLangLexer>();
    auto *xbParser = syntaxContext.getParser<xblang::XBLangParser>();
    assert(xbParser && "invalid xblang parser");
    auto &parser =
        syntaxContext.getOrRegisterParser<GPUParser>(context, *xbParser);
    auto &dynParser = syntaxContext.getDynParser();
    parser.setDynamicParser(&dynParser);
    dynParser.registerCombinator(kw, xblang::TypeInfo::get<xblang::xlg::Stmt>(),
                                 &parser, GPUParser::invokeStmt);
  }
};
} // namespace

void ::gpu::registerSyntaxInterface(mlir::DialectRegistry &registry) {
  registry.insert<mlir::gpu::GPUDialect>();
  registry.addExtension(
      +[](mlir::MLIRContext *ctx, mlir::gpu::GPUDialect *dialect) {
        dialect->addInterfaces<GPUSyntax>();
      });
}

void ::gpu::registerSyntaxInterface(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  registerSyntaxInterface(registry);
  context.appendDialectRegistry(registry);
}

//===----------------------------------------------------------------------===//
// GPUParser
//===----------------------------------------------------------------------===//

GPUParser::GPUParser(::xblang::XBContext *ctx,
                     ::xblang::xlg::XLGBuilder &builder,
                     ::xblang::SourceManager &srcManager, Lexer &lexer)
    : Base(srcManager, lexer), ::xblang::xlg::XLGBuilderRef(builder) {}

mlir::Operation *GPUParser::createLaunchOp(
    mlir::Location loc, mlir::SmallVectorImpl<mlir::Value> &bsz,
    mlir::SmallVectorImpl<mlir::Value> &gsz, xblang::InsertionBlock &block) {
  block.restorePoint();
  Value c1{};
  Type indexTy = builder.getIndexType();
  for (Value &value : bsz) {
    if (!value)
      value = c1 ? c1 : (c1 = builder.create<index::ConstantOp>(loc, 1));
    else
      value = builder.create<xblang::xbg::FromXLGExpr>(
          loc, indexTy, TypeAttr::get(indexTy), value);
  }
  for (Value &value : gsz) {
    if (!value)
      value = c1 ? c1 : (c1 = builder.create<index::ConstantOp>(loc, 1));
    else
      value = builder.create<xblang::xbg::FromXLGExpr>(
          loc, indexTy, TypeAttr::get(indexTy), value);
  }
  auto op = builder.create<mlir::gpu::LaunchOp>(loc, gsz[0], gsz[1], gsz[2],
                                                bsz[0], bsz[1], bsz[2]);
  Block &body = op.getBody().front();
  body.getOperations().splice(body.end(), block.getBlock()->getOperations());
  auto grd = builder.guard(builder, &body);
  builder.create<mlir::gpu::TerminatorOp>(loc);
  return op;
}

mlir::Value GPUParser::createIdDimOp(mlir::Location loc, int kind,
                                     mlir::gpu::Dimension dim) {
  Value id;
  if (kind == 0)
    id = builder.create<mlir::gpu::ThreadIdOp>(loc, dim);
  else if (kind == 1)
    id = builder.create<mlir::gpu::BlockIdOp>(loc, dim);
  else if (kind == 2)
    id = builder.create<mlir::gpu::BlockDimOp>(loc, dim);
  else
    id = builder.create<mlir::gpu::GridDimOp>(loc, dim);
  return builder.create<xblang::xbg::ToXLGExpr>(
      loc, builder.getConceptClass<xblang::xlg::Expr>(),
      TypeAttr::get(id.getType()), id);
}

void registerRegionCombinators(GPUParser &parser) {
  parser.registerCombinator<xblang::xlg::Expr>("threadIdx",
                                               GPUParser::invokeId);
  parser.registerCombinator<xblang::xlg::Expr>("blockIdx", GPUParser::invokeId);
  parser.registerCombinator<xblang::xlg::Expr>("blockDim", GPUParser::invokeId);
  parser.registerCombinator<xblang::xlg::Expr>("gridDim", GPUParser::invokeId);
}
