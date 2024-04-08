//===- Syntax.cpp - Defines the XBLang syntax --------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the xblang syntax.
//
//===----------------------------------------------------------------------===//

#include "xblang/Lang/XBLang/Syntax/Syntax.h"
#include "xblang/Dialect/XBLang/IR/Enums.h"
#include "xblang/Dialect/XBLang/IR/Type.h"
#include "xblang/Interfaces/Syntax.h"
#include "xblang/Support/Format.h"
#include "xblang/Support/LLVM.h"
#include "xblang/Syntax/ParsingCombinators.h"
#include "xblang/Syntax/SyntaxContext.h"
#include "xblang/XLG/Concepts.h"
#include "llvm/Support/Error.h"

using namespace ::xblang;
using namespace ::xblang::xbg;

#include "xblang/Lang/XBLang/Syntax/XBLangParser.cpp.inc"

using namespace xblang;
using namespace xblang::syntax;

namespace {
//===----------------------------------------------------------------------===//
// XGBSemaInterface
//===----------------------------------------------------------------------===//
class XGBSyntaxInterface : public xblang::SyntaxDialectInterface {
public:
  using xblang::SyntaxDialectInterface::SyntaxDialectInterface;

  void populateSyntax(XBContext *context,
                      SyntaxContext &syntaxContext) const override {
    syntaxContext.getOrRegisterLexer<XBLangLexer>();
    auto &parser = syntaxContext.getOrRegisterParser<XBLangParser>(
        context, (Block *){nullptr});
    auto &dynParser = syntaxContext.getDynParser();
    parser.setDynamicParser(&dynParser);
    dynParser.registerCombinator("", TypeInfo::get<xlg::Decl>(), &parser,
                                 XBLangParser::invokeTopDecl);
    dynParser.registerCombinator("", TypeInfo::get<xlg::Expr>(), &parser,
                                 XBLangParser::invokeExpr);
    dynParser.registerCombinator("", TypeInfo::get<xlg::Stmt>(), &parser,
                                 XBLangParser::invokeStmt);
    dynParser.registerCombinator("", TypeInfo::get<xlg::Type>(), &parser,
                                 XBLangParser::invokeType);
  }
};
} // namespace

void xblang::xbg::registerXBGSyntaxInterface(mlir::DialectRegistry &registry) {
  registry.insert<XBGDialect>();
  registry.addExtension(+[](mlir::MLIRContext *ctx, XBGDialect *dialect) {
    dialect->addInterfaces<XGBSyntaxInterface>();
  });
}

void xblang::xbg::registerXBGSyntaxInterface(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  registerXBGSyntaxInterface(registry);
  context.appendDialectRegistry(registry);
}

//===----------------------------------------------------------------------===//
// XBLangParser
//===----------------------------------------------------------------------===//

XBLangParser::XBLangParser(XBContext *ctx, Block *block,
                           SourceManager &srcManager, Lexer &lexer)
    : Base(srcManager, lexer), xlg::XLGBuilder(ctx) {}

::xblang::syntax::ParsingStatus
XBLangParser::parseState(const SourceState &state) {
  auto bLoc = getLoc();
  lex(state);
  auto grd = guard(*this);
  PureParsingStatus status;
  while (lex.isValid() && !lex.isEndOfFile()) {
    status = parseTopDecl(true);
    if (!status.isSuccess())
      break;
  }
  auto tok = getTok();
  if (!tok.isEndOfFile() && !status.isAnError())
    emitError(tok.getLoc(), "failed to parse the full state");
  return success(bLoc);
}

namespace {
inline int operatorPrecedence(BinaryOperator kind) {
  switch (kind) {
  case BinaryOperator::Assign:
    return 100;
  case BinaryOperator::CompoundAdd:
  case BinaryOperator::CompoundSub:
  case BinaryOperator::CompoundMul:
  case BinaryOperator::CompoundDiv:
  case BinaryOperator::CompoundMod:
  case BinaryOperator::CompoundAnd:
  case BinaryOperator::CompoundOr:
  case BinaryOperator::CompoundBinaryAnd:
  case BinaryOperator::CompoundBinaryOr:
  case BinaryOperator::CompoundBinaryXor:
  case BinaryOperator::CompoundLShift:
  case BinaryOperator::CompoundRShift:
    return 200;
  case BinaryOperator::Ternary:
    return 400;
  case BinaryOperator::Or:
    return 500;
  case BinaryOperator::And:
    return 600;
  case BinaryOperator::BinaryOr:
    return 610;
  case BinaryOperator::BinaryXor:
    return 620;
  case BinaryOperator::BinaryAnd:
    return 630;
  case BinaryOperator::Less:
  case BinaryOperator::Greater:
  case BinaryOperator::LEQ:
  case BinaryOperator::GEQ:
    return 700;
  case BinaryOperator::Spaceship:
  case BinaryOperator::Equal:
  case BinaryOperator::NEQ:
    return 800;
  case BinaryOperator::LShift:
  case BinaryOperator::RShift:
    return 900;
  case BinaryOperator::Add:
  case BinaryOperator::Sub:
    return 1000;
  case BinaryOperator::Mul:
  case BinaryOperator::Div:
  case BinaryOperator::Mod:
    return 1100;
  default:
    return -1;
  }
}
} // namespace

syntax::ParseResult<mlir::Value> XBLangParser::parseExpr(bool emitErrors) {
  auto lhs = parseTopExpr(emitErrors);
  if (!lhs.isSuccess())
    return lhs;
  auto result = parseBinaryExpr(emitErrors, 0, lhs);
  if (result.isSuccess() && getTok() == Lexer::Colon)
    return parseRangeExpr(emitErrors, result.get());
  return result;
}

void XBLangParser::invokeExpr(::xblang::syntax::ParserBase *parserPtr,
                              ::xblang::SourceState &state,
                              ::xblang::syntax::ParsingStatus &status,
                              bool emitErrors) {
  auto &parser = static_cast<XBLangParser &>(*parserPtr);
  auto &result =
      static_cast<::xblang::syntax::ParseResult<::mlir::Value> &>(status);
  auto grd = parser.getGuard();
  parser.setState(state);
  result = parser.parseExpr(emitErrors);
  state = parser.lex.getState();
  state.restore(parser.getTok().getLoc());
}

syntax::ParseResult<mlir::Value>
XBLangParser::parseBinaryExpr(bool emitErrors, int exprPrecedence,
                              syntax::ParseResult<mlir::Value> lhs) {
  if (!lhs.isSuccess() || !lhs.get())
    return error(lhs.getLoc());
  while (true) {
    Token tok = getTok();
    auto bop = XBLangLexer::toBinaryOp(XBLangLexer::getToken(tok.getTok()));
    int token_precedence = operatorPrecedence(bop);
    if (token_precedence < exprPrecedence)
      return success(std::move(lhs), getLoc());
    consume();
    auto rhs = parseTopExpr(emitErrors);
    if (!rhs.isSuccess())
      return error(rhs.getLoc());
    auto nextOp = XBLangLexer::toBinaryOp(XBLangLexer::getToken(getTok()));
    int next_precedence = operatorPrecedence(nextOp);
    if (token_precedence == next_precedence &&
        (bop == BinaryOperator::Assign || isCompoundOp(bop))) {
      rhs = parseBinaryExpr(emitErrors, token_precedence, rhs);
      if (!rhs.get()) {
        emitError(rhs.getLoc(), "invalid expression");
        return error(rhs.getLoc());
      }
    } else if (token_precedence < next_precedence) {
      rhs = parseBinaryExpr(emitErrors, token_precedence + 1, rhs);
      if (!rhs.get()) {
        emitError(rhs.getLoc(), "invalid expression");
        return error(rhs.getLoc());
      }
    }
    if (bop == BinaryOperator::Ternary) {
      // Build a ternary expression.
      tok = getTok();
      if (tok == Lexer::Colon)
        consume();
      else {
        emitError(tok.getLoc(), "expected a `:`");
        return error<Value>(tok.getLoc());
      }
      auto tmp = parseTopExpr(emitErrors);
      if (!tmp.isSuccess())
        return tmp;
      auto falseValue = parseBinaryExpr(emitErrors, 0, tmp);
      if (!falseValue.isSuccess())
        return falseValue;
      lhs.get() = create<SelectExpr>(lhs.get().getLoc(),
                                     getConceptClass<SelectExprCep>(), nullptr,
                                     lhs.get(), rhs.get(), falseValue.get());
    } else {
      // Build the binary expression.
      lhs.get() =
          create<BinOpExpr>(lhs.get().getLoc(), getConceptClass<BinOpExprCep>(),
                            nullptr, bop, lhs.get(), rhs.get());
    }
  }
  return error<mlir::Value>(getLoc());
}

ParseResult<Value> XBLangParser::parseTopExpr(bool emitErrors) {
  auto bLoc = getLoc();
  auto grd = getGuard(emitErrors);
  auto tok = getTok();
  SmallVector<UnaryOperator> uops;
  UnaryOperator uop;
  while ((uop = Lexer::toUnaryOp(tok.getTok())) != UnaryOperator::None) {
    uops.push_back(uop);
    tok = consume();
  }
  ParseResult<Value> expr = parseCoreExpr(emitErrors);
  if (!expr.isSuccess())
    return expr;
  tok = getTok();
  // Parse postfix decorators.
  while (lex.isValid() && !lex.isEndOfFile()) {
    switch (tok.getTok()) {
    // Parse a call expression.
    case Lexer::LParen: {
      consume();
      ParseResult<SmallVector<Value>> arguments = parseCommaListExpr(false);
      if (arguments.isAnError()) {
        emitError(bLoc, "failed to parse the call expression");
        return error<Value>(bLoc);
      }
      if ((tok = getTok()) != Lexer::RParen) {
        emitError(bLoc, "expected a `)`");
        return error<Value>(tok.getLoc());
      }
      tok = consume();
      expr.get() =
          create<CallExpr>(expr.get().getLoc(), getConceptClass<CallExprCep>(),
                           nullptr, expr.get(), arguments.get());
      continue;
    }
    // Parse an array expression.
    case Lexer::LBracket: {
      consume();
      ParseResult<SmallVector<Value>> arguments = parseCommaListExpr(false);
      if (arguments.isAnError()) {
        emitError(bLoc, "failed to parse the array expression");
        return error<Value>(bLoc);
      }
      if ((tok = getTok()) != Lexer::RBracket) {
        emitError(bLoc, "expected a `)`");
        return error<Value>(tok.getLoc());
      }
      tok = consume();
      expr.get() = create<ArrayExpr>(expr.get().getLoc(),
                                     getConceptClass<ArrayExprCep>(), nullptr,
                                     expr.get(), arguments.get());
      continue;
    }
    default:
      break;
    }
    break;
  }
  for (auto uop : llvm::reverse(uops))
    expr.get() =
        create<UnaryExpr>(expr.get().getLoc(), getConceptClass<UnaryExprCep>(),
                          nullptr, uop, expr.get());
  grd.release();
  return success<Value>(std::move(expr.get()), bLoc);
}

mlir::Value XBLangParser::getIntExpr(const Token &tok) {
  Lexer::IntLiteralInfo info = Lexer::getIntInfo(tok.getTok());
  StringRef literal = tok.getSpelling();
  if (info.radix != 10)
    literal = literal.drop_front(2);
  if (info.width > 8)
    literal = literal.drop_back(3);
  else if (info.width <= 8 && info.width > 0)
    literal = literal.drop_back(2);
  else if (info.signedness == IntegerType::Unsigned)
    literal = literal.drop_back(1);
  Type type;
  if (info.width != 0)
    type = IntegerType::get(getContext(), info.width, info.signedness);
  else {
    if (info.signedness == IntegerType::Signed)
      type = getIntegerType(64, true);
    else
      type = getIndexType();
    info.width = 64;
  }
  auto numBits = llvm::APInt::getBitsNeeded(literal, info.radix);
  llvm::APInt value(numBits < info.width ? info.width : numBits, literal,
                    info.radix);
  numBits = value.getBitWidth();
  if (numBits > info.width) {
    value = value.trunc(info.width);
    emitWarning(
        tok.getLoc(),
        fmt("literal was truncated to fit its type. It required '{0}' bits, "
            "but the type has '{1}' bits. The literal in memory is: {2}",
            numBits, info.width, value.getZExtValue()));
  }
  return create<xbg::ConstExpr>(getLoc(tok), getConceptClass<IntExpr>(),
                                TypeAttr::get(type),
                                getIntegerAttr(type, value));
}

mlir::Value XBLangParser::getFloatExpr(const Token &tok) {
  mlir::FloatType type{};
  StringRef spelling = tok.getSpelling();
  auto *semantics = &llvm::APFloat::IEEEdouble();
  if (tok != Lexer::FloatLiteral)
    spelling = spelling.drop_back(3);
  if (tok == Lexer::FloatLiteralf128)
    spelling = spelling.drop_back(1);
  switch (tok.getTok()) {
  case Lexer::FloatLiteralf16:
    semantics = &llvm::APFloat::IEEEhalf();
    type = mlir::FloatType::getF32(context);
    break;
  case Lexer::FloatLiteralf32:
    semantics = &llvm::APFloat::IEEEsingle();
    type = mlir::FloatType::getF32(context);
    break;
  case Lexer::FloatLiteral:
  case Lexer::FloatLiteralf64:
    semantics = &llvm::APFloat::IEEEdouble();
    type = mlir::FloatType::getF64(context);
    break;
  case Lexer::FloatLiteralf128:
    semantics = &llvm::APFloat::IEEEquad();
    type = mlir::FloatType::getF128(context);
    break;
  default:
    break;
  }
  llvm::APFloat value(*semantics);
  llvm::Expected<llvm::APFloatBase::opStatus> status =
      value.convertFromString(spelling, llvm::APFloat::rmTowardZero);
  if (!status) {
    emitError(tok.getLoc(), "invalid floating point literal.");
    llvm::consumeError(status.takeError());
    return nullptr;
  }
  return create<xbg::ConstExpr>(getLoc(tok), getConceptClass<FloatExpr>(),
                                TypeAttr::get(type), getFloatAttr(type, value));
}

mlir::Value XBLangParser::getBoolExpr(const Token &tok) {
  return create<xbg::ConstExpr>(getLoc(tok), getConceptClass<BoolExpr>(),
                                TypeAttr::get(getI1Type()),
                                getBoolAttr(tok == Lexer::True));
}

mlir::Value XBLangParser::getStringExpr(const Token &tok) {
  return create<xbg::ConstExpr>(getLoc(tok), getConceptClass<StringExpr>(),
                                TypeAttr::get(getType<xb::StringType>()),
                                getStringAttr(tok.getSpelling()));
}

mlir::Value XBLangParser::getNullExpr() { return nullptr; }

ParseResult<::mlir::Value>
XBLangParser::parseQualifiedIdentifierExpr(bool emitErrors) {
  auto grd = getGuard(emitErrors);
  auto tok = getTok();
  auto bLoc = tok.getLoc();
  bool isRoot = tok == Lexer::Namespace;
  // Check for a qualified search from the top.
  if (isRoot)
    tok = consume();
  // Emit an error if we are not parsing an identifier.
  if (tok != Lexer::Identifier) {
    if (emitErrors)
      emitError(tok.getLoc(), "expected an identifier");
    return error<::mlir::Value>(bLoc);
  }
  llvm::StringRef root;
  llvm::SmallVector<mlir::FlatSymbolRefAttr> ids;
  // Consume all identifiers.
  while (tok == Lexer::Identifier) {
    if (root.empty())
      root = tok.getSpelling();
    else
      ids.push_back(
          mlir::FlatSymbolRefAttr::get(getContext(), tok.getSpelling()));
    tok = consume();
    if (tok == Lexer::Namespace)
      tok = consume();
    else
      break;
  }
  Value base = create<RefExpr>(
      getLoc(bLoc), getConceptClass<RefExpr::ConceptType>(), nullptr,
      mlir::SymbolRefAttr::get(getContext(), root, ids), tok == Lexer::LParen);
  if (tok != Lexer::Dot) {
    grd.release();
    return success<::mlir::Value>(std::move(base), bLoc);
  }
  llvm::SmallVector<mlir::FlatSymbolRefAttr> memberIds;
  tok = consume();
  while (tok == Lexer::Identifier) {
    memberIds.push_back(
        mlir::FlatSymbolRefAttr::get(getContext(), tok.getSpelling()));
    tok = consume();
    if (tok == Lexer::Dot)
      tok = consume();
    else
      break;
  }
  MemberRefExpr expr{};
  for (auto id : memberIds) {
    base = expr =
        create<MemberRefExpr>(getLoc(bLoc), getConceptClass<MemberRefExprCep>(),
                              nullptr, id, false, base);
  }
  tok.getTok();
  expr.setDelayedResolution(tok == Lexer::LParen);
  grd.release();
  return success<::mlir::Value>(std::move(base), bLoc);
}

ParseResult<::mlir::Value>
XBLangParser::parseQualifiedIdentifierType(bool emitErrors) {
  auto grd = getGuard(emitErrors);
  auto tok = getTok();
  auto bLoc = tok.getLoc();
  bool isRoot = tok == Lexer::Namespace;
  // Check for a qualified search from the top.
  if (isRoot)
    tok = consume();
  // Emit an error if we are not parsing an identifier.
  if (tok != Lexer::Identifier) {
    if (emitErrors)
      emitError(tok.getLoc(), "expected an identifier");
    return error<::mlir::Value>(bLoc);
  }
  llvm::StringRef root;
  llvm::SmallVector<mlir::FlatSymbolRefAttr> ids;
  // Consume all identifiers.
  while (tok == Lexer::Identifier) {
    if (root.empty())
      root = tok.getSpelling();
    else
      ids.push_back(
          mlir::FlatSymbolRefAttr::get(getContext(), tok.getSpelling()));
    tok = consume();
    if (tok == Lexer::Namespace)
      tok = consume();
    else
      break;
  }
  grd.release();
  return success<::mlir::Value>(
      create<RefExprType>(getLoc(bLoc),
                          getConceptClass<RefExprType::ConceptType>(), nullptr,
                          mlir::SymbolRefAttr::get(getContext(), root, ids)),
      bLoc);
}

syntax::ParseResult<::mlir::Value> XBLangParser::parseRangeExpr(bool emitErrors,
                                                                Value expr) {
  auto grd = getGuard(emitErrors);
  auto tok = getTok();
  if (tok == Lexer::Colon)
    tok = consume();
  else {
    if (emitErrors)
      emitError(tok.getLoc(), "expected a semicolon");
    return error(tok.getLoc());
  }
  BinaryOperator stepOp = BinaryOperator::firstBinOp,
                 cmpOp = BinaryOperator::firstBinOp;
  BinaryOperatorAttr stepAttr{}, cmpAttr{};
  switch (tok.getTok()) {
  case Lexer::Plus:
    stepOp = BinaryOperator::Add;
    break;
  case Lexer::Dash:
    stepOp = BinaryOperator::Sub;
    break;
  case Lexer::Asterisk:
    stepOp = BinaryOperator::Mul;
    break;
  case Lexer::Slash:
    stepOp = BinaryOperator::Div;
    break;
  case Lexer::LShift:
    stepOp = BinaryOperator::LShift;
    break;
  case Lexer::RShift:
    stepOp = BinaryOperator::RShift;
    break;
  case Lexer::Less:
    cmpOp = BinaryOperator::Less;
    break;
  case Lexer::Greater:
    cmpOp = BinaryOperator::Greater;
    break;
  case Lexer::LEq:
    cmpOp = BinaryOperator::LEQ;
    break;
  case Lexer::GEq:
    cmpOp = BinaryOperator::GEQ;
    break;
  case Lexer::Equality:
    cmpOp = BinaryOperator::Equal;
    break;
  case Lexer::NEq:
    cmpOp = BinaryOperator::NEQ;
    break;
  default:
    break;
  }
  if (stepOp != BinaryOperator::firstBinOp)
    consume(), stepAttr = getAttr<BinaryOperatorAttr>(stepOp);
  if (cmpOp != BinaryOperator::firstBinOp)
    consume(), cmpAttr = getAttr<BinaryOperatorAttr>(cmpOp);
  // Parse a expression.
  auto lhs = parseTopExpr(emitErrors);
  if (!lhs.isSuccess())
    return lhs;
  auto expr1 = parseBinaryExpr(emitErrors, 0, lhs);
  if (!expr1.isSuccess())
    return expr1;
  tok = getTok();
  if (tok == Lexer::Colon && cmpOp == BinaryOperator::firstBinOp) {
    tok = consume();
  } else {
    grd.release();
    return success<Value>(create<RangeExpr>(expr.getLoc(),
                                            getConceptClass<RangeExprCep>(),
                                            nullptr, cmpAttr, expr, expr1.get(),
                                            stepAttr, nullptr),
                          tok.getLoc());
  }
  switch (tok.getTok()) {
  case Lexer::Less:
    cmpOp = BinaryOperator::Less;
    break;
  case Lexer::Greater:
    cmpOp = BinaryOperator::Greater;
    break;
  case Lexer::LEq:
    cmpOp = BinaryOperator::LEQ;
    break;
  case Lexer::GEq:
    cmpOp = BinaryOperator::GEQ;
    break;
  case Lexer::Equality:
    cmpOp = BinaryOperator::Equal;
    break;
  case Lexer::NEq:
    cmpOp = BinaryOperator::NEQ;
    break;
  default:
    break;
  }
  if (cmpOp != BinaryOperator::firstBinOp)
    consume(), cmpAttr = getAttr<BinaryOperatorAttr>(cmpOp);
  // Parse a expression.
  lhs = parseTopExpr(emitErrors);
  if (!lhs.isSuccess())
    return lhs;
  auto expr2 = parseBinaryExpr(emitErrors, 0, lhs);
  if (!expr2.isSuccess())
    return expr2;
  grd.release();
  return success<Value>(
      create<RangeExpr>(expr.getLoc(), getConceptClass<RangeExprCep>(), nullptr,
                        cmpAttr, expr, expr2.get(), stepAttr, expr1.get()),
      tok.getLoc());
}

mlir::Operation *
XBLangParser::createRangeForStmt(Location loc, InsertionBlock &&block,
                                 SmallVector<RangeLoopInfo> &ranges) {
  block.restorePoint();
  SmallVector<Value, 8> iterators;
  SmallVector<Value, 8> rangeExprs;
  for (auto &loop : ranges) {
    if (loop.type != nullptr) {
      auto vd = create<VarDecl>(getLoc(loop.id), getConceptClass<VarDeclCep>(),
                                loop.id.getSpelling(), nullptr, nullptr,
                                loop.type, nullptr);
      iterators.push_back(
          create<ValueRefExpr>(vd.getLoc(), getConceptClass<ValueRefExprCep>(),
                               nullptr, vd.getConceptClass()));
    } else
      iterators.push_back(
          create<RefExpr>(getLoc(loop.id), getConceptClass<RefExprCep>(),
                          nullptr, getSymRef(loop.id.getSpelling())));
    rangeExprs.push_back(loop.range);
  }
  RangeForStmt op = create<RangeForStmt>(
      loc, getConceptClass<RangeForStmtCep>(), iterators, rangeExprs);
  op.getBodyRegion().push_back(block.release());
  return op;
}

xlg::TemplateOp
XBLangParser::makeDeclTemplate(const SourceLocation &loc, Operation *decl,
                               InsertionBlock &block,
                               ArrayRef<xlg::TemplateParam> parameters) {
  block.restorePoint();
  ConceptType type;
  {
    auto grd = guard(*this);
    if (auto op = dyn_cast<FuncDecl>(decl)) {
      type = op.getConceptClass().getType();
      setInsertionPoint(op.getBody(0), op.getBody(0)->begin());
    } else if (auto op = dyn_cast<ObjectDecl>(decl)) {
      type = op.getConceptClass().getType();
      setInsertionPoint(op.getBody(0), op.getBody(0)->begin());
    }
    for (auto &param : parameters) {
      auto blockArg =
          block.getBlock()->addArgument(param.conceptClass, param.loc);
      auto *cep = param.conceptClass.getConceptClass().getConcept();
      if (isa<TemplateTypeCep>(cep)) {
        create<TemplateType>(param.loc, param.conceptClass, param.identifier,
                             nullptr, nullptr, blockArg, param.init);
      } else if (isa<TemplateExprCep>(cep)) {
        create<TemplateExpr>(param.loc, param.conceptClass, param.identifier,
                             nullptr, nullptr, blockArg, param.init);
      }
    }
  }
  return createTemplate(getLoc(loc), type, block.release(), parameters, decl);
}
