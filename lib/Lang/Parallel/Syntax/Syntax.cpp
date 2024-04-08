//===- Syntax.cpp - Defines the Par syntax -----------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the par syntax.
//
//===----------------------------------------------------------------------===//

#include "xblang/Lang/Parallel/Syntax/Syntax.h"
#include "xblang/Dialect/Parallel/IR/Dialect.h"
#include "xblang/Dialect/Parallel/IR/Parallel.h"
#include "xblang/Dialect/XBLang/IR/Enums.h"
#include "xblang/Dialect/XBLang/IR/Type.h"
#include "xblang/Interfaces/Syntax.h"
#include "xblang/Lang/XBLang/Syntax/Syntax.h"
#include "xblang/Lang/XBLang/XLG/XBGDialect.h"
#include "xblang/Support/Format.h"
#include "xblang/Support/LLVM.h"
#include "xblang/Syntax/ParsingCombinators.h"
#include "xblang/Syntax/SyntaxContext.h"
#include "xblang/XLG/Concepts.h"
#include "llvm/Support/Error.h"

using namespace ::xblang;
using namespace ::xblang::par;

#include "xblang/Lang/Parallel/Syntax/ParParser.cpp.inc"

using namespace xblang::syntax;

namespace {
//===----------------------------------------------------------------------===//
// ParSemaInterface
//===----------------------------------------------------------------------===//
class ParSyntaxInterface : public xblang::SyntaxDialectInterface {
public:
  using xblang::SyntaxDialectInterface::SyntaxDialectInterface;

  void populateSyntax(XBContext *context,
                      SyntaxContext &syntaxContext) const override {
    StringRef kw = "par";
    syntaxContext.getOrRegisterLexer<XBLangLexer>();
    auto *xbParser = syntaxContext.getParser<XBLangParser>();
    assert(xbParser && "invalid xblang parser");
    auto &parser =
        syntaxContext.getOrRegisterParser<ParParser>(context, *xbParser);
    auto &dynParser = syntaxContext.getDynParser();
    parser.setDynamicParser(&dynParser);
    dynParser.registerCombinator(kw, TypeInfo::get<xlg::Stmt>(), &parser,
                                 ParParser::invokeCombinatorStmt);
    dynParser.registerCombinator(kw, TypeInfo::get<xlg::Expr>(), &parser,
                                 ParParser::invokeCombinatorExpr);
  }
};
} // namespace

void xblang::par::registerParSyntaxInterface(mlir::DialectRegistry &registry) {
  registry.insert<xblang::xbg::XBGDialect>();
  registry.insert<mlir::par::ParDialect>();
  registry.addExtension(
      +[](mlir::MLIRContext *ctx, mlir::par::ParDialect *dialect) {
        dialect->addInterfaces<ParSyntaxInterface>();
      });
}

void xblang::par::registerParSyntaxInterface(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  registerParSyntaxInterface(registry);
  context.appendDialectRegistry(registry);
}

//===----------------------------------------------------------------------===//
// XBLangParser
//===----------------------------------------------------------------------===//

ParParser::ParParser(XBContext *ctx, xlg::XLGBuilder &builder,
                     SourceManager &srcManager, Lexer &lexer)
    : Base(srcManager, lexer), xlg::XLGBuilderRef(builder) {}
