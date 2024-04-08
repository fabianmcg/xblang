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
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "omp/Extension.h"
#include "xblang/Lang/XBLang/Syntax/Syntax.h"
#include "xblang/Lang/XBLang/XLG/XBGDecl.h"
#include "xblang/Lang/XBLang/XLG/XBGType.h"
#include "xblang/Syntax/ParsingCombinators.h"
#include "xblang/Syntax/SyntaxContext.h"
#include "xblang/XLG/Concepts.h"
#include "xblang/XLG/IR/XLGDialect.h"
#include "llvm/Support/Error.h"

using namespace ::omp;

#include "omp/OMPParser.cpp.inc"

using namespace xblang::syntax;
using namespace mlir;

namespace {
//===----------------------------------------------------------------------===//
// OMPSyntax
//===----------------------------------------------------------------------===//
class OMPSyntax : public xblang::SyntaxDialectInterface {
public:
  using xblang::SyntaxDialectInterface::SyntaxDialectInterface;

  void populateSyntax(xblang::XBContext *context,
                      xblang::SyntaxContext &syntaxContext) const override {
    llvm::StringRef kw = "omp";
    syntaxContext.getOrRegisterLexer<xblang::XBLangLexer>();
    auto *xbParser = syntaxContext.getParser<xblang::XBLangParser>();
    assert(xbParser && "invalid xblang parser");
    auto &parser =
        syntaxContext.getOrRegisterParser<OMPParser>(context, *xbParser);
    auto &dynParser = syntaxContext.getDynParser();
    parser.setDynamicParser(&dynParser);
    dynParser.registerCombinator(kw, xblang::TypeInfo::get<xblang::xlg::Stmt>(),
                                 &parser, OMPParser::invokeStmt);
  }
};
} // namespace

void ::omp::registerSyntaxInterface(mlir::DialectRegistry &registry) {
  registry.insert<mlir::omp::OpenMPDialect>();
  registry.addExtension(
      +[](mlir::MLIRContext *ctx, mlir::omp::OpenMPDialect *dialect) {
        dialect->addInterfaces<OMPSyntax>();
      });
}

void ::omp::registerSyntaxInterface(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  registerSyntaxInterface(registry);
  context.appendDialectRegistry(registry);
}

//===----------------------------------------------------------------------===//
// OMPParser
//===----------------------------------------------------------------------===//

OMPParser::OMPParser(::xblang::XBContext *ctx,
                     ::xblang::xlg::XLGBuilder &builder,
                     ::xblang::SourceManager &srcManager, Lexer &lexer)
    : Base(srcManager, lexer), ::xblang::xlg::XLGBuilderRef(builder) {}

mlir::Operation *
OMPParser::createParallelOp(mlir::Location loc, xblang::InsertionBlock &block,
                            mlir::ArrayRef<DataSharingInfo> sharingClauses) {
  block.restorePoint();
  auto op = builder.create<mlir::omp::ParallelOp>(loc);
  Block *body = block.release();
  auto grd = builder.guard(builder);
  builder.setInsertionPointToEnd(body);
  builder.create<mlir::omp::TerminatorOp>(loc);
  builder.setInsertionPointToStart(body);
  for (auto clause : sharingClauses) {
    if (clause.kind == clause.Shared)
      continue;
    Location loc = getLoc(clause.tok);
    Value refExpr{}, typeTy{};
    refExpr = builder.create<xblang::xbg::RefExpr>(
        loc, getConceptClass<xblang::xbg::RefExprCep>(), nullptr,
        builder.getSymRef(clause.tok.getSpelling()));
    typeTy = builder.create<xblang::xbg::TypeOf>(
        loc, getConceptClass<xblang::xbg::TypeOfCep>(), nullptr, refExpr);
    typeTy = builder.create<xblang::xbg::RemoveReferenceType>(
        loc, getConceptClass<xblang::xbg::RemoveReferenceTypeCep>(), nullptr,
        typeTy);
    if (clause.kind != clause.FirstPrivate)
      refExpr = nullptr;
    builder.create<xblang::xbg::VarDecl>(
        loc, getConceptClass<xblang::xbg::VarDeclCep>(),
        clause.tok.getSpelling(), nullptr, nullptr, typeTy, refExpr);
  }
  op.getRegion().push_back(body);
  return op;
}
