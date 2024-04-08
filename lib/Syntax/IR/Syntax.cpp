//===- SyntaxDialect.cpp - Syntax dialect -----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Syntax dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "xblang/Support/Format.h"
#include "xblang/Syntax/IR/SyntaxDialect.h"
#include "xblang/Syntax/Utils/SDTUtils.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace xblang::syntaxgen;

//===----------------------------------------------------------------------===//
// CharClass
//===----------------------------------------------------------------------===//
void CharClass::computeUSR() {
  for (auto r : charRanges) {
    if (r.isChar())
      usr += xblang::utfToString(r.lowerBound);
    else
      usr += xblang::utfToString(r.lowerBound) + "-" +
             xblang::utfToString(r.upperBound);
  }
}

bool CharClass::operator==(const CharClass &other) const {
  return usr == other.usr;
}

std::string CharClass::toString() const {
  std::string repr;
  llvm::raw_string_ostream os(repr);
  std::string usr;
  auto toCodePoint = [](uint32_t c) -> std::string {
    if (c == '-')
      return "\\-";
    return xblang::utfToString(c);
  };
  for (auto r : charRanges) {
    if (r.isChar())
      usr += toCodePoint(r.lowerBound);
    else
      usr += toCodePoint(r.lowerBound) + "-" + toCodePoint(r.upperBound);
  }
  llvm::printEscapedString(usr, os);
  return repr;
}

std::optional<CharClass> CharClass::fromString(SourceState &state) {
  SourceLocation loc;
  std::string error;
  struct SDTLexerCommon::Char tok;
  auto consume = [&]() {
    tok = !state.isEndOfFile() ? SDTLexerCommon::consumeChar(state, loc, error)
                               : SDTLexerCommon::Char::invalid();
  };
  consume();
  if (tok.isInvalid())
    return std::nullopt;
  CharClass charClass;
  while (state.isValid() && !tok.isInvalid()) {
    uint32_t l = tok.character;
    if (tok.isUTF() || (tok.isControl() && tok.character != '-')) {
      consume();
      if (tok.isControl() && tok.character == '-') {
        consume();
        if (tok.isControl() && tok.character == '-')
          return std::nullopt;
        charClass.insert(l, tok.character);
        consume();
        continue;
      }
      charClass.insert(l);
      continue;
    }
    return std::nullopt;
  }
  charClass.computeUSR();
  return charClass;
}

llvm::hash_code xblang::syntaxgen::hash_value(const CharClass &value) {
  return llvm::hash_value(value.getUSR());
}

namespace mlir {
template <>
struct FieldParser<CharClass> {
  static FailureOr<CharClass> parse(AsmParser &parser) {
    std::string str;
    if (succeeded(parser.parseString(&str))) {
      xblang::SourceState state(str);
      if (auto charClass = CharClass::fromString(state);
          charClass != std::nullopt)
        return *charClass;
      else
        return parser.emitError(parser.getCurrentLocation(),
                                "invalid character class");
    }
    return failure();
  }
};
} // namespace mlir

//===----------------------------------------------------------------------===//
// Literal Attr
//===----------------------------------------------------------------------===//

namespace {
ParseResult parseLiteral(mlir::AsmParser &odsParser, uint32_t &code) {
  std::string str;
  if (succeeded(odsParser.parseOptionalString(&str))) {
    if (str.size() == 1)
      code = str[0];
    else
      return odsParser.emitError(odsParser.getCurrentLocation())
             << "literal must be a single character";
  } else if (odsParser.parseInteger(code)) {
    return failure();
  }
  return success();
}

void printLiteral(mlir::AsmPrinter &printer, uint32_t code) {
  if (code < 128 && llvm::isASCII(static_cast<char>(code))) {
    char str[2] = {static_cast<char>(code), 0};
    printer.printString(StringRef(str));
  } else {
    printer << code;
  }
}
} // namespace

//===----------------------------------------------------------------------===//
// Macro Op
//===----------------------------------------------------------------------===//

void MacroOp::build(OpBuilder &builder, OperationState &odsState,
                    StringRef name, size_t numArgs) {
  auto exprTy = builder.getType<ExprType>();
  SmallVector<Type, 8> args(numArgs, exprTy);
  buildWithEntryBlock(builder, odsState, name,
                      builder.getFunctionType(args, TypeRange({exprTy})), {},
                      args);
}

mlir::ParseResult MacroOp::parse(mlir::OpAsmParser &parser,
                                 mlir::OperationState &result) {
  auto fnBuilder =
      [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
         llvm::ArrayRef<mlir::Type> results,
         mlir::function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };
  return mlir::function_interface_impl::parseFunctionOp(
      parser, result, false, getFunctionTypeAttrName(result.name), fnBuilder,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void MacroOp::print(mlir::OpAsmPrinter &p) {
  mlir::function_interface_impl::printFunctionOp(
      p, *this, false, getFunctionTypeAttrName(), getArgAttrsAttrName(),
      getResAttrsAttrName());
}

//===----------------------------------------------------------------------===//
// Rule Op
//===----------------------------------------------------------------------===//

namespace {
ParseResult parseRuleType(OpAsmParser &parser, TypeAttr &ty) {
  auto &builder = parser.getBuilder();
  ty = TypeAttr::get(builder.getFunctionType(
      TypeRange(), TypeRange({builder.getType<ExprType>()})));
  return success();
}

void printRuleType(OpAsmPrinter &printer, Operation *, TypeAttr ty) {}
} // namespace

void RuleOp::build(OpBuilder &builder, OperationState &odsState,
                   StringRef name) {
  RuleOp::build(builder, odsState, name,
                builder.getFunctionType(
                    TypeRange(), TypeRange({builder.getType<ExprType>()})),
                nullptr, nullptr, nullptr);
}

//===----------------------------------------------------------------------===//
// And Op
//===----------------------------------------------------------------------===//

void AndOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), "and");
}

OpFoldResult AndOp::fold(FoldAdaptor adaptor) {
  if (auto lhs = getLHS().getDefiningOp(); lhs && isa<EmptyStringOp>(lhs))
    return getRHS();
  else if (auto rhs = getRHS().getDefiningOp(); rhs && isa<EmptyStringOp>(rhs))
    return getLHS();
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Or Op
//===----------------------------------------------------------------------===//

void OrOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), "or");
}

OpFoldResult OrOp::fold(FoldAdaptor adaptor) {
  return getLHS() == getRHS() ? getLHS() : nullptr;
}

//===----------------------------------------------------------------------===//
// Empty String Op
//===----------------------------------------------------------------------===//

void EmptyStringOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), "eps");
}

OpFoldResult EmptyStringOp::fold(FoldAdaptor adaptor) {
  return UnitAttr::get(getContext());
}

//===----------------------------------------------------------------------===//
// Terminal Op
//===----------------------------------------------------------------------===//

void TerminalOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  if (auto named = dyn_cast<LexTerminalAttr>(getTerminalAttr()))
    setNameFn(getResult(), named.getIdentifier().getValue());
  else
    setNameFn(getResult(), "terminal");
}

OpFoldResult TerminalOp::fold(FoldAdaptor adaptor) { return getTerminal(); }

//===----------------------------------------------------------------------===//
// NonTerminal Op
//===----------------------------------------------------------------------===//

void NonTerminalOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), getNonTerminal());
}

OpFoldResult NonTerminalOp::fold(FoldAdaptor adaptor) {
  if (auto attr = getDynamicAttr())
    return attr;
  return getNonTerminalAttr();
}

//===----------------------------------------------------------------------===//
// Call Op
//===----------------------------------------------------------------------===//

void CallOp::build(OpBuilder &builder, OperationState &odsState,
                   StringRef name) {
  CallOp::build(builder, odsState, builder.getAttr<FlatSymbolRefAttr>(name),
                ValueRange());
}

void CallOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), "call");
}

Operation::operand_range CallOp::getArgOperands() {
  return (*this)->getOperands();
}

MutableOperandRange CallOp::getArgOperandsMutable() {
  return MutableOperandRange(*this);
}

CallInterfaceCallable CallOp::getCallableForCallee() {
  return (*this)->getAttrOfType<mlir::SymbolRefAttr>("callee");
}

void CallOp::setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
  (*this)->setAttr("callee", callee.get<mlir::SymbolRefAttr>());
}

//===----------------------------------------------------------------------===//
// Zero Or More Op
//===----------------------------------------------------------------------===//

void ZeroOrMoreOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), "zom");
}

OpFoldResult ZeroOrMoreOp::fold(FoldAdaptor adaptor) {
  Operation *expr = getExpr().getDefiningOp();
  if (expr && isa<ZeroOrMoreOp>(expr))
    return getExpr();
  if (expr && isa<EmptyStringOp>(expr))
    return getExpr();
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Action Op
//===----------------------------------------------------------------------===//

void MetadataOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), "md");
}

OpFoldResult MetadataOp::fold(FoldAdaptor adaptor) {
  if (!getCodeActionAttr() && !getName())
    return getExpr();
  return nullptr;
}

LogicalResult MetadataOp::verify() {
  Operation *expr = getExpr().getDefiningOp();
  if (AndOp andExpr = dyn_cast_or_null<AndOp>(expr))
    for (auto users : andExpr.getResult().getUsers())
      if (auto op = dyn_cast<AndOp>(users))
        return emitError("`md_nodes` cannot be in-between of `and` ops");
  if (OrOp orExpr = dyn_cast_or_null<OrOp>(expr))
    for (auto users : orExpr.getResult().getUsers())
      if (auto op = dyn_cast<OrOp>(users))
        return emitError("`md_nodes` cannot be in-between of `or` ops");
  return success();
}

//===----------------------------------------------------------------------===//
// Switch Op
//===----------------------------------------------------------------------===//

void SwitchOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), "switch");
}

OpFoldResult SwitchOp::fold(FoldAdaptor adaptor) {
  auto alternatives = getAlternatives();
  return alternatives.size() == 1 ? alternatives.front() : nullptr;
}

//===----------------------------------------------------------------------===//
// Sequence Op
//===----------------------------------------------------------------------===//

void SequenceOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), "seq");
}

OpFoldResult SequenceOp::fold(FoldAdaptor adaptor) {
  auto alternatives = getAlternatives();
  return alternatives.size() == 1 ? alternatives.front() : nullptr;
}

//===----------------------------------------------------------------------===//
// Any Op
//===----------------------------------------------------------------------===//

void AnyOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), "any");
}

OpFoldResult AnyOp::fold(FoldAdaptor adaptor) {
  auto alternatives = getAlternatives();
  return alternatives.size() == 1 ? alternatives.front() : nullptr;
}

//===----------------------------------------------------------------------===//
// Lex dialect
//===----------------------------------------------------------------------===//

namespace {
struct SyntaxInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  bool isLegalToInline(Operation *op, Region *, bool, IRMapping &) const final {
    return true;
  }

  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return true;
  }

  void handleTerminator(Operation *op, ValueRange valuesToRepl) const final {
    auto returnOp = cast<ReturnOp>(op);
    assert(returnOp->getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp->getOperands()))
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }

  Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                       Type resultType,
                                       Location conversionLoc) const final {
    return nullptr;
  }
};

struct SyntaxASM : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (auto action = dyn_cast<CodeActionAttr>(attr)) {
      os << "code";
      return AliasResult::OverridableAlias;
    }
    return AliasResult::NoAlias;
  }
};
} // namespace

void SyntaxDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "xblang/Syntax/IR/SyntaxOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "xblang/Syntax/IR/SyntaxOpsTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "xblang/Syntax/IR/SyntaxOpsAttributes.cpp.inc"
      >();
  addInterfaces<SyntaxInlinerInterface>();
  addInterfaces<SyntaxASM>();
}

#include "xblang/Syntax/IR/SyntaxOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "xblang/Syntax/IR/SyntaxOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "xblang/Syntax/IR/SyntaxOpsAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "xblang/Syntax/IR/SyntaxOpsTypes.cpp.inc"

#include "xblang/Syntax/IR/SyntaxOpsEnums.cpp.inc"
