//===- XLGDialect.cpp - XLG dialect -----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the XLG dialect.
//
//===----------------------------------------------------------------------===//

#include "xblang/XLG/IR/XLGDialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "xblang/Basic/ContextDialect.h"
#include "xblang/Support/Format.h"
#include "xblang/XLG/Concepts.h"
#include "xblang/XLG/IR/XLGTraits.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace xblang::xlg;

//===----------------------------------------------------------------------===//
// XLG dialect
//===----------------------------------------------------------------------===//

void XLGDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "xblang/XLG/IR/XLGOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "xblang/XLG/IR/XLGOpsTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "xblang/XLG/IR/XLGOpsAttributes.cpp.inc"
      >();
  registerConcepts(
      getContext()->getLoadedDialect<XBContextDialect>()->getContext());
}

namespace {
ParseResult parseConceptType(AsmParser &parser,
                             FailureOr<xblang::ConceptContainer> &c) {
  StringRef kw, kwConstruct;
  if (failed(parser.parseKeyword(&kw)))
    return failure();
  auto dialect =
      parser.getContext()->getLoadedDialect<xblang::XBContextDialect>();
  if (!dialect)
    return parser.emitError(parser.getCurrentLocation(),
                            "XBlang context is not loaded.");
  xblang::XBContext &context = dialect->getContext();
  if (failed(parser.parseOptionalColon())) {
    c = xblang::ConceptContainer(context.getConcept("", kw));
    return success();
  }
  if (parser.parseColon())
    return failure();
  if (failed(parser.parseKeyword(&kwConstruct)))
    return failure();
  c = xblang::ConceptContainer(context.getConcept(kw, kwConstruct));
  return success();
}

void printConceptType(AsmPrinter &printer, xblang::ConceptContainer c) {
  if (StringRef dialect = c.getDialect(); !dialect.empty())
    printer << dialect << "::";
  printer << c.getIdentifier();
}
} // namespace

ConceptType xblang::xlg::getConceptClass(Concept *cep) {
  if (mlir::MLIRContext *ctx = Concept::getMLIRContext(cep))
    return ConceptType::get(ctx, ConceptContainer(cep));
  return {};
}

mlir::ParseResult xblang::xlg::parseXLGConcept(mlir::OpAsmParser &parser,
                                               mlir::Type &type, Concept *cep) {
  if (!cep)
    return parser.emitError(parser.getCurrentLocation(), "invalid concept");
  if (succeeded(parser.parseOptionalLess())) {
    FailureOr<xblang::ConceptContainer> c;
    if (failed(parseConceptType(parser, c)))
      return failure();
    if (parser.parseGreater())
      return failure();
    type = ConceptType::get(parser.getContext(), *c);
  } else
    type = ConceptType::get(parser.getContext(), xblang::ConceptContainer(cep));
  return success();
}

void xblang::xlg::printXLGConcept(mlir::OpAsmPrinter &printer, ConceptType type,
                                  Concept *cep) {
  if (type.getConceptClass() != cep) {
    printer << "<";
    printConceptType(printer, type.getConceptClass());
    printer << ">";
  }
}

mlir::ParseResult xblang::xlg::parseXLGConcept(OpAsmParser &parser,
                                               TypeAttr &type, Concept *cep) {
  if (!cep)
    return parser.emitError(parser.getCurrentLocation(), "invalid concept");
  if (succeeded(parser.parseOptionalLess())) {
    FailureOr<xblang::ConceptContainer> c;
    if (failed(parseConceptType(parser, c)))
      return failure();
    if (parser.parseGreater())
      return failure();
    type = TypeAttr::get(ConceptType::get(parser.getContext(), *c));
  } else
    type = TypeAttr::get(
        ConceptType::get(parser.getContext(), xblang::ConceptContainer(cep)));
  return success();
}

void xblang::xlg::printXLGConcept(OpAsmPrinter &printer, TypeAttr typeAttr,
                                  Concept *cep) {
  auto type = dyn_cast<ConceptType>(typeAttr.getValue());
  assert(type && "invalid type attribute");
  if (type.getConceptClass() != cep) {
    printer << "<";
    printConceptType(printer, type.getConceptClass());
    printer << ">";
  }
}

//===----------------------------------------------------------------------===//
// XLG Template Op
//===----------------------------------------------------------------------===//

void TemplateOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       llvm::StringRef name, mlir::FunctionType type,
                       mlir::Block *body) {
  state.addAttribute(TemplateOp::getFunctionTypeAttrName(state.name),
                     TypeAttr::get(type));
  Region *bodyRegion = state.addRegion();
  if (body)
    bodyRegion->push_back(body);
  for (mlir::Type input : type.getInputs())
    body->addArgument(input, state.location);
  state.getOrAddProperties<Properties>().setSymId(builder.getStringAttr(name));
}

mlir::ParseResult TemplateOp::parse(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {
  return failure();
}

void TemplateOp::print(mlir::OpAsmPrinter &p) {
  auto fnTy = getFunctionType();
  p << " @" << getSymId();
  if (auto usr = getUSR())
    p << "[" << usr << "]";
  p << "(";
  if (getBodyRegion().empty()) {
    llvm::interleaveComma(getBodyRegion().getArguments(), p,
                          [&](BlockArgument arg) { p.printOperand(arg); });
  } else {
    size_t i = 0;
    llvm::interleaveComma(fnTy.getInputs(), p, [&](mlir::Type arg) {
      p << "%arg" << i++ << " : " << arg;
    });
  }
  p << ") ";
  ArrayRef<mlir::Type> results = fnTy.getResults();
  if (results.size() == 1)
    p << "-> " << results.front();
  if (results.size() > 1) {
    p << "-> (";
    llvm::interleaveComma(results, p, [&](mlir::Type arg) { p << arg; });
    p << ") ";
  }
  if (!getBodyRegion().empty()) {
    p.printRegion(getBodyRegion(), false);
  }
}

xblang::SymbolProperties TemplateOp::getSymbolProps() {
  return xblang::SymbolProperties::Template;
}

xblang::SymbolTableKind TemplateOp::getSymbolTableKind() {
  return xblang::SymbolTableKind::Ordered;
}

xblang::SymbolTableProperties TemplateOp::getSymbolTableProps() {
  return xblang::SymbolTableProperties::None;
}

#include "xblang/XLG/IR/XLGOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "xblang/XLG/IR/XLGOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "xblang/XLG/IR/XLGOpsAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "xblang/XLG/IR/XLGOpsTypes.cpp.inc"
