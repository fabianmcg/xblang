//===- Syntax.cpp - Defines the meta extension syntax ------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the meta extension syntax.
//
//===----------------------------------------------------------------------===//

#include "xblang/Lang/Meta/Syntax/Syntax.h"
#include "mlir/AsmParser/AsmParser.h"
#include "xblang/Interfaces/Syntax.h"
#include "xblang/Lang/XBLang/Syntax/Syntax.h"
#include "xblang/Lang/XBLang/XLG/XBGType.h"
#include "xblang/Support/Format.h"
#include "xblang/Support/LLVM.h"
#include "xblang/Syntax/ParsingCombinators.h"
#include "xblang/Syntax/SyntaxContext.h"
#include "xblang/XLG/Concepts.h"
#include "xblang/XLG/IR/XLGDialect.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SourceMgr.h"

using namespace ::xblang;
using namespace ::xblang::meta;

#include "xblang/Lang/Meta/Syntax/MetaParser.cpp.inc"

using namespace xblang::syntax;

namespace {
//===----------------------------------------------------------------------===//
// MetaSyntaxInterface
//===----------------------------------------------------------------------===//
class MetaSyntaxInterface : public xblang::SyntaxDialectInterface {
public:
  using xblang::SyntaxDialectInterface::SyntaxDialectInterface;

  void populateSyntax(XBContext *context,
                      SyntaxContext &syntaxContext) const override {
    StringRef kw = "mlir";
    syntaxContext.getOrRegisterLexer<XBLangLexer>();
    auto *xbParser = syntaxContext.getParser<XBLangParser>();
    assert(xbParser && "invalid xblang parser");
    auto &parser =
        syntaxContext.getOrRegisterParser<MetaParser>(context, *xbParser);
    auto &dynParser = syntaxContext.getDynParser();
    parser.setDynamicParser(&dynParser);
    dynParser.registerCombinator(kw, TypeInfo::get<xlg::Attr>(), &parser,
                                 MetaParser::invokeAttr);
    dynParser.registerCombinator(kw, TypeInfo::get<xlg::Decl>(), &parser,
                                 MetaParser::invokeTop);
    dynParser.registerCombinator(kw, TypeInfo::get<xlg::Expr>(), &parser,
                                 MetaParser::invokeInline);
    dynParser.registerCombinator(kw, TypeInfo::get<xlg::Type>(), &parser,
                                 MetaParser::invokeType);
  }
};
} // namespace

void xblang::meta::registerMetaSyntaxInterface(
    mlir::DialectRegistry &registry) {
  registry.addExtension(+[](mlir::MLIRContext *ctx, xlg::XLGDialect *dialect) {
    dialect->addInterfaces<MetaSyntaxInterface>();
  });
}

void xblang::meta::registerMetaSyntaxInterface(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  registerMetaSyntaxInterface(registry);
  context.appendDialectRegistry(registry);
}

//===----------------------------------------------------------------------===//
// XBLangParser
//===----------------------------------------------------------------------===//

MetaParser::MetaParser(XBContext *ctx, xlg::XLGBuilder &builder,
                       SourceManager &srcManager, Lexer &lexer)
    : Base(srcManager, lexer), xlg::XLGBuilderRef(builder) {}

::mlir::Attribute MetaParser::getAttr(const SourceLocation &loc,
                                      StringRef literal) {
  size_t numRead = 0;
  literal = literal.ltrim().rtrim();
  Attribute attr =
      mlir::parseAttribute(literal, builder.getContext(), nullptr, &numRead);
  if (!attr)
    emitError(loc, "invalid attr");
  return attr;
}

Value MetaParser::getInlineValue(const SourceLocation &loc,
                                 ArrayRef<std::pair<Token, StringRef>> args,
                                 std::pair<StringRef, StringRef> ret,
                                 StringRef literal) {
  SmallString<128> code;
  for (auto arg : args) {
    StringRef valName = arg.first.getSpelling();
    code += fmt("%__priv{0} = xbg.ref_expr @{0}\n", valName);
    code += fmt("%{0} = xbg.from_xlg_expr <xbg::ref_expr> %__priv{0} : {1} "
                "{{type = {1}}\n",
                valName, arg.second);
  }
  code += literal.ltrim().rtrim().str();
  code.c_str();
  llvm::SourceMgr mgr;
  mgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBuffer(code, "mlir_inline", true), loc.loc);
  mlir::ParserConfig config(builder.getContext(), false);
  std::unique_ptr<Block> block(new Block());
  if (failed(mlir::parseAsmSourceFile(mgr, block.get(), config)))
    return nullptr;
  xlg::ConceptType type =
      !ret.second.empty()
          ? builder.getConceptClass(
                builder.getXBContext()->getConcept(ret.first, ret.second))
          : builder.getConceptClass<xlg::Expr>();
  auto op = builder.create<xlg::RegionOp>(getLoc(loc), type);
  op.getBodyRegion().push_back(block.release());
  return op;
}

syntax::PureParsingStatus MetaParser::getInlineOp(const SourceLocation &loc,
                                                  StringRef literal) {
  SmallString<128> code;
  code += literal.ltrim().rtrim().str();
  code.c_str();
  llvm::SourceMgr mgr;
  mgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBuffer(code, "mlir_inline", true), loc.loc);
  mlir::ParserConfig config(builder.getContext(), false);
  std::unique_ptr<Block> block(new Block());
  if (failed(mlir::parseAsmSourceFile(mgr, block.get(), config)))
    return PureParsingStatus::Error;
  Block *curBlock = builder.getBlock();
  curBlock->getOperations().splice(curBlock->end(), block->getOperations());
  return PureParsingStatus::Success;
}

Value MetaParser::getType(Location loc, StringRef literal) {
  size_t numRead = 0;
  literal = literal.ltrim().rtrim();
  Type type = mlir::parseType(literal, builder.getContext(), &numRead);
  if (!type) {
    mlir::emitError(loc, "invalid type");
    return nullptr;
  }
  return builder.create<xbg::BuiltinType>(
      loc, builder.getConceptClass<xlg::Type>(), type);
}
