//===- ParserGen.cpp - Parser Generator ----------------------------*-
// C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the TableGen parser generator.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/TableGen/GenInfo.h"
#include "xblang/Support/Format.h"
#include "xblang/Syntax/IR/SyntaxDialect.h"
#include "xblang/Syntax/ParserGen/SDTranslation.h"
#include "xblang/Syntax/Transforms/Passes.h"
#include "xblang/TableGen/Lexer.h"
#include "xblang/TableGen/Parser.h"
#include "xblang/TableGen/Util.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

#include <stack>

#include "TemplateEngine.h"

using namespace xblang;
using namespace xblang::tablegen;
using namespace xblang::syntaxgen;
using namespace xblang::syntaxgen::parser;

namespace {
llvm::cl::OptionCategory genCat("Options for op definition generators");
}

typedef enum {
  CodeGenStage,
  ProcessSyntaxStage,
  AnalizeSyntaxStage,
  LastStage,
} PipelineStage;

static llvm::cl::opt<std::string>
    parserFilter("parser-name",
                 llvm::cl::desc("Name of the parser to generate"),
                 llvm::cl::cat(genCat), llvm::cl::init(""));

static llvm::cl::opt<PipelineStage> pipelineStage(
    llvm::cl::desc("Parser gen final stage:"), llvm::cl::cat(genCat),
    "gen-parser-opt-stage",
    llvm::cl::values(
        clEnumValN(CodeGenStage, "codegen", "Initial code gen"),
        clEnumValN(ProcessSyntaxStage, "syntax", "Canonicalize the syntax"),
        clEnumValN(AnalizeSyntaxStage, "analyze", "Analyze the syntax")),
    llvm::cl::init(LastStage));

namespace {
using Environment = TextTemplate::Environment;

/// Rule generator.
struct ProductionGen {
  ProductionGen(Production production, RuleOp productionOp, Parser parser,
                ParserOp parserModule)
      : production(production), productionOp(productionOp), parser(parser),
        parserModule(parserModule) {}

  /// Emits the parser declaration.
  mlir::LogicalResult emitDecl(llvm::raw_ostream &os, const Environment &env);

  /// Emits the parser definition.
  mlir::LogicalResult emitDef(llvm::raw_ostream &os, const Environment &env);

  /// Returns the production record.
  Production getProduction() const { return production; }

private:
  struct GenContext {
    StringRef controlPoint;
    StringRef guard;
  };

  enum {
    SwitchCtrl = 1,
    AnyCtrl,
    ZeroOrMoreCtrl,
    KindMaskCtrl = 0xF,
    NullableCtrl = 16,
    LocalCtrl = 32
  };

  // Visits a switch operation.
  mlir::LogicalResult visitSwitch(SwitchOp op, llvm::raw_ostream &os);
  // Visits an any operation.
  mlir::LogicalResult visitAny(AnyOp op, llvm::raw_ostream &os);
  // Visits a seq operation.
  mlir::LogicalResult visitSequence(SequenceOp op, llvm::raw_ostream &os);
  // Visits a zero or more operation.
  mlir::LogicalResult visitZeroOrMore(ZeroOrMoreOp op, llvm::raw_ostream &os);
  // Visits a terminal operation.
  mlir::LogicalResult visitTerminal(TerminalOp op, llvm::raw_ostream &os,
                                    StringRef name);
  // Visits a terminal operation.
  mlir::LogicalResult visitEmptyString(EmptyStringOp op, llvm::raw_ostream &os);
  // Visits a non-terminal operation.
  mlir::LogicalResult visitNonTerminal(NonTerminalOp op, llvm::raw_ostream &os,
                                       StringRef name);
  // Visits a non-terminal operation.
  mlir::LogicalResult visitMetadata(MetadataOp op, llvm::raw_ostream &os);
  // Visits a generic operation.
  mlir::LogicalResult visit(mlir::Operation *op, llvm::raw_ostream &os,
                            StringRef name = "");
  // Emits a lexing terminal condition.
  std::string emitLexTerminal(mlir::Attribute attr, StringRef sym, bool match);
  // Emits a first set condition.
  std::string emitFirstSet(mlir::ArrayAttr attr, StringRef sym);
  // Emits a control point.
  void emitControlPoint(llvm::raw_ostream &os, const std::string &label,
                        const std::string &guard, int ctrlKind);
  /// Production being emitted.
  Production production;
  /// Production op being emitted.
  RuleOp productionOp;
  /// Parser being processed.
  Parser parser;
  /// Parser module.
  ParserOp parserModule;
  /// Context stack.
  std::stack<GenContext> contextStack;
  size_t switchCount = {};
  size_t anyCount = {};
  size_t zomCount = {};
  size_t combinatorCount = {};
};

//===----------------------------------------------------------------------===//
// Parser generator
//===----------------------------------------------------------------------===//
class ParserGen {
public:
  /// Generates the MLIR parser module.
  static std::optional<ParserGen> genModule(Parser parser,
                                            mlir::OpBuilder builder,
                                            xblang::SourceManager &srcMgr);

  /// Emits the parser declaration.
  mlir::LogicalResult emitDecl(llvm::raw_ostream &os);

  /// Emits the parser definition.
  mlir::LogicalResult emitDef(llvm::raw_ostream &os);

private:
  ParserGen(Parser parser);
  /// Parser being processed.
  Parser parser;
  /// Parser module.
  ParserOp parserModule;
  /// List of productions in the parser.
  SmallVector<ProductionGen> productions;
};

//===----------------------------------------------------------------------===//
// Global generator
//===----------------------------------------------------------------------===//
class Generator {
public:
  Generator(const llvm::RecordKeeper &records);

  /// Emits the parser declarations.
  mlir::LogicalResult emitDecl(llvm::raw_ostream &os);

  /// Emits the parser definitions.
  mlir::LogicalResult emitDef(llvm::raw_ostream &os);

  /// Dump the parsers.
  mlir::LogicalResult dump(llvm::raw_ostream &os);

private:
  /// Initializes the generator.
  void init(const llvm::RecordKeeper &records);

  /// Run the MLIR pass pipeline
  mlir::LogicalResult runPipeline(PipelineStage stage = LastStage);

  /// MLIR context.
  mlir::MLIRContext context;
  /// Source manager.
  xblang::SourceManager sourceManager;
  /// parsers being generated.
  SmallVector<ParserGen> parsers;
  /// Module containing all parsers.
  mlir::OwningOpRef<mlir::ModuleOp> module;
};
} // namespace

//===----------------------------------------------------------------------===//
// Global generator
//===----------------------------------------------------------------------===//

Generator::Generator(const llvm::RecordKeeper &records)
    : context(), sourceManager(&context) {
  context.loadDialect<xblang::syntaxgen::SyntaxDialect>();
  sourceManager.registerDiagnosticsHandler();
  init(records);
}

void Generator::init(const llvm::RecordKeeper &records) {
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  auto inputFile = records.getInputFilename();
  mlir::OpBuilder builder(module.getBody(0), module.getBody(0)->end());
  for (llvm::Record *v : records.getAllDerivedDefinitions("Parser")) {
    if (auto par = Parser::castOrNull(v)) {
      if (!parserFilter.empty() && parserFilter != par->getIdentifier())
        continue;
      auto parser = ParserGen::genModule(*par, builder, sourceManager);
      if (!parser) {
        llvm::PrintFatalError(v, "failed to process the parser");
        return;
      }
      parsers.push_back(std::move(*parser));
    }
  }
  this->module = std::move(module);
}

mlir::LogicalResult Generator::runPipeline(PipelineStage stage) {
  if (!module)
    return mlir::failure();
  mlir::PassManager pm(&context);
  mlir::OpPassManager &nestedPM = pm.nest<ParserOp>();
  if (stage >= ProcessSyntaxStage)
    nestedPM.addPass(createProcessSyntax(ProcessSyntaxOptions{false}));
  if (stage >= AnalizeSyntaxStage)
    nestedPM.addPass(createAnalyzeSyntax());
  return pm.run(*module);
}

mlir::LogicalResult Generator::dump(llvm::raw_ostream &os) {
  if (!module)
    return mlir::failure();
  if (mlir::failed(runPipeline(pipelineStage)))
    return mlir::failure();
  module.get().print(os);
  return mlir::success();
}

mlir::LogicalResult Generator::emitDecl(llvm::raw_ostream &os) {
  if (!module)
    return mlir::failure();
  if (mlir::failed(runPipeline(pipelineStage)))
    return mlir::failure();
  for (auto &par : parsers)
    if (mlir::failed(par.emitDecl(os)))
      return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult Generator::emitDef(llvm::raw_ostream &os) {
  if (!module)
    return mlir::failure();
  if (mlir::failed(runPipeline(pipelineStage)))
    return mlir::failure();
  for (auto &par : parsers)
    if (mlir::failed(par.emitDef(os)))
      return mlir::failure();
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Lexer generator
//===----------------------------------------------------------------------===//

ParserGen::ParserGen(Parser parser) : parser(parser) {}

std::optional<ParserGen> ParserGen::genModule(Parser parser,
                                              mlir::OpBuilder builder,
                                              xblang::SourceManager &srcMgr) {
  TerminalMap map(parser.getLexer(), parser.getDefaultToken());
  ParserGen parserGen(parser);
  mlir::OpBuilder::InsertionGuard guard(builder);
  // Create the parser module.
  auto parseModule = builder.create<ParserOp>(
      builder.getUnknownLoc(), builder.getStringAttr(parser.getStartSymbol()),
      builder.getStringAttr(parser.getIdentifier()), nullptr);
  // Insert the block of the module and set the builder.
  builder.setInsertionPointToEnd(&parseModule.getBodyRegion().emplaceBlock());
  parserGen.parserModule = parseModule;
  mlir::SymbolTable parseTable(parseModule);
  SDTranslator sdTranslator(srcMgr, map, parseTable);
  for (auto macro : parser.getMacros()) {
    auto id = macro.getIdentifier();
    if (!sdTranslator
             .parseMacro(srcMgr.createSource(macro.getExpr(), id),
                         macro.getArgs(), id, *parseModule.getBody(0))
             .isSuccess())
      return {};
  }
  for (auto rule : parser.getProductions()) {
    auto id = rule.getIdentifier();
    syntax::ParseResult<mlir::Operation *> parseResult =
        sdTranslator.parseProduction(srcMgr.createSource(rule.getRule(), id),
                                     id, *parseModule.getBody(0));
    if (!parseResult.isSuccess())
      return {};
    parserGen.productions.push_back(ProductionGen(
        rule, mlir::cast<RuleOp>(parseResult.get()), parser, parseModule));
  }
  return std::move(parserGen);
}

mlir::LogicalResult ParserGen::emitDecl(llvm::raw_ostream &os) {

  TemplateEngine tmpl = TemplateEngine::make(StringRef(R"(
//===----------------------------------------------------------------------===//
// Parser $parserName
//===----------------------------------------------------------------------===//
class $parserName : 
  public ::xblang::syntax::ParserMixin<$parserName, $LexerNamespace::$lexerName,
                                       ::xblang::syntax::PackratContext>$TraitList {
public:
  using Base = ::xblang::syntax::ParserMixin<$parserName,
                                             $LexerNamespace::$lexerName,
                                             ::xblang::syntax::PackratContext>;
  using Base::ParserMixin;
  typedef enum {$ProductionEnum} ProductionID;
$Productions
$ExtraClassDeclarations
};
)"));
  Environment env;
  TextTemplate::insert(env, "parserName",
                       StrTemplate::make(parser.getIdentifier()));
  TextTemplate::insert(env, "cppNamespace",
                       StrTemplate::make(parser.getCppNamespace()));
  std::string productionsStr, prodEnum;
  llvm::raw_string_ostream pos(productionsStr);
  for (auto &gen : productions) {
    if (mlir::failed(gen.emitDecl(pos, env)))
      return mlir::failure();
    prodEnum += llvm::convertToCamelFromSnakeCase(
                    gen.getProduction().getIdentifier(), true) +
                "Prod,\n";
  }
  tmpl.insert("ExtraClassDeclarations",
              TemplateEngine::make(parser.getExtraClassDeclarations()));
  tmpl.insert("Productions", StrTemplate::make(productionsStr));
  tmpl.insert("ProductionEnum", StrTemplate::make(prodEnum));
  tmpl.insert("lexerName",
              StrTemplate::make(parser.getLexer().getIdentifier()));
  tmpl.insert("LexerNamespace",
              StrTemplate::make(parser.getLexer().getCppNamespace()));
  std::string traitList;
  for (auto &trait : parser.getTraits()) {
    auto nt = llvm::dyn_cast<NativeTrait>(&trait);
    if (!nt) {
      llvm::PrintFatalError(&parser.getDef(), "invalid trait");
      return mlir::failure();
    }
    traitList += ", public " + nt->getFullyQualifiedTraitName();
  }
  tmpl.insert("TraitList", TemplateEngine::make(traitList));
  llvm::SmallVector<StringRef, 2> namespaces;
  llvm::SplitString(parser.getCppNamespace(), namespaces, "::");
  for (StringRef ns : namespaces)
    os << "namespace " << ns << " {\n";
  os << tmpl.compile(env);
  for (StringRef ns : llvm::reverse(namespaces))
    os << "} // namespace " << ns << "\n";
  return mlir::success();
}

mlir::LogicalResult ParserGen::emitDef(llvm::raw_ostream &os) {
  if (!parser.getImplement())
    return mlir::success();
  TemplateEngine tmpl = TemplateEngine::make(StringRef(R"(
//===----------------------------------------------------------------------===//
// Begin parser $parserName
//===----------------------------------------------------------------------===//
$ExtraClassDefinitions
$Productions
//===----------------------------------------------------------------------===//
// End parser $parserName
//===----------------------------------------------------------------------===//
)"));
  Environment env;
  TextTemplate::insert(env, "parserName",
                       StrTemplate::make(parser.getIdentifier()));
  TextTemplate::insert(env, "lexer", StrTemplate::make("lex"));
  TextTemplate::insert(env, "cppNamespace",
                       StrTemplate::make(parser.getCppNamespace()));
  std::string productionsStr;
  llvm::raw_string_ostream pos(productionsStr);
  for (auto &gen : productions)
    if (mlir::failed(gen.emitDef(pos, env)))
      return mlir::failure();
  tmpl.insert("ExtraClassDefinitions",
              TemplateEngine::make(parser.getExtraClassDefinitions()));
  tmpl.insert("Productions", StrTemplate::make(productionsStr));
  llvm::SmallVector<StringRef, 2> namespaces;
  llvm::SplitString(parser.getCppNamespace(), namespaces, "::");
  for (StringRef ns : namespaces)
    os << "namespace " << ns << " {\n";
  os << tmpl.compile(env);
  for (StringRef ns : llvm::reverse(namespaces))
    os << "} // namespace " << ns << "\n";
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Rule generator
//===----------------------------------------------------------------------===//

mlir::LogicalResult ProductionGen::emitDecl(llvm::raw_ostream &os,
                                            const Environment &env) {
  TemplateEngine tmpl = TemplateEngine::make(R"(
/// Parsing rule: $ruleName
::xblang::syntax::ParseResult<$resultTy> parse$ruleName($RuleArgs);
static void invoke$ruleName(::xblang::syntax::ParserBase* parser,
  ::xblang::SourceState& state, ::xblang::syntax::ParsingStatus& status,
  $RuleArgs);
)");
  std::string code;
  llvm::raw_string_ostream cos(code);
  cos << "bool emitErrors";
  for (auto [i, param] :
       llvm::enumerate(production.getArguments().getCppList())) {
    if (!param) {
      llvm::PrintFatalError(&production.getDef(), "invalid arguments");
      return mlir::failure();
    }
    std::string name =
        param->getName() ? param->getName()->str() : "_arg" + std::to_string(i);
    auto defVal = param->getDefaultValue();
    std::string init = defVal ? " = " + defVal->str() : "";
    cos << fmt(", {0} {1}{2}", param->getCppType(), name, init);
  }
  tmpl.insert("RuleArgs", StrTemplate::make(code));
  tmpl.insert("resultTy", TemplateEngine::make(production.getReturnType()));
  tmpl.insert("ruleName", StrTemplate::make(llvm::convertToCamelFromSnakeCase(
                              production.getIdentifier(), true)));
  os << tmpl.compile(env);
  return mlir::success();
}

mlir::LogicalResult ProductionGen::emitDef(llvm::raw_ostream &os,
                                           const Environment &env) {
  if (!production.getImplement())
    return mlir::success();
  Environment locelEnv(env);
  std::string code;
  llvm::raw_string_ostream cos(code);
  contextStack.push(GenContext{"Exit", "_ctx.guard"});
  // Emit the rule:
  ReturnOp ret =
      dyn_cast_or_null<ReturnOp>(productionOp.getBody(0)->getTerminator());
  if (mlir::failed(visit(ret.getExpr().getDefiningOp(), cos)))
    return mlir::failure();
  contextStack.pop();
  // Emit the production:
  TemplateEngine tmpl = TemplateEngine::make(R"(
::xblang::syntax::ParseResult<$resultTy> $parserName::parse$ruleName($RuleArgs) {
  if (auto cache = getCache<$resultTy>($lexer, ${ruleName}Prod))
    return *cache;
  auto _ctx = getProductionContext("failed to parse production: `${ruleName}`", $emitErrors);
  auto _tok = getTok();
  (void)_tok;
  std::optional<$resultTy> _res = std::nullopt;
  {
${Production}// Exit ${ruleName}
  }
Exit:
  if ($ctx.guard.isAnError() && !$ctx.diag)
     $diag = emitError($bLoc, "failed to parse production: `${ruleName}`");
  (void)ctrlCheck($ctx, $ctx.guard, $ctx.guard, CtrlProduction, ${NullableRule}, false, $emitErrors);
  return saveResult(std::move($res), $ctx.guard.getStatus(), $bLoc, getBuf(), ${ruleName}Prod);
}
void $parserName::invoke$ruleName(::xblang::syntax::ParserBase* parserPtr,
  ::xblang::SourceState& state, ::xblang::syntax::ParsingStatus& status, $RuleArgs) {
  auto &parser = static_cast<$parserName&>(*parserPtr);
  auto &result = static_cast<::xblang::syntax::ParseResult<$resultTy>&>(status);
  auto grd = parser.getGuard();
  parser.setState(state);
  result = parser.parse$ruleName(_emitErrors$argsExpansion);
  state = parser.lex.getState();
  state.restore(parser.getTok().getLoc());
}
)");
  tmpl.insert("Production", TemplateEngine::make(code));
  code.clear();
  cos << "bool _emitErrors";
  std::string argsExpansion;
  for (auto [i, param] :
       llvm::enumerate(production.getArguments().getCppList())) {
    if (!param) {
      llvm::PrintFatalError(&production.getDef(), "invalid arguments");
      return mlir::failure();
    }
    std::string name =
        param->getName() ? param->getName()->str() : "_arg" + std::to_string(i);
    cos << fmt(", {0} {1}", param->getCppType(), name);
    argsExpansion += ", " + name;
  }
  tmpl.insert("RuleArgs", StrTemplate::make(code));
  tmpl.insert(
      "NullableRule",
      StrTemplate::make(productionOp->getAttr("nullable") ? "true" : "false"));
  TextTemplate::insert(locelEnv, "resultTy",
                       TemplateEngine::make(production.getReturnType()));
  TextTemplate::insert(locelEnv, "ruleName",
                       StrTemplate::make(llvm::convertToCamelFromSnakeCase(
                           production.getIdentifier(), true)));
  TextTemplate::insert(locelEnv, "tok", StrTemplate::make("_tok"));
  TextTemplate::insert(locelEnv, "bLoc", StrTemplate::make("_ctx.bLoc"));
  TextTemplate::insert(locelEnv, "diag", StrTemplate::make("_ctx"));
  TextTemplate::insert(locelEnv, "res", StrTemplate::make("_res"));
  TextTemplate::insert(locelEnv, "ctx", StrTemplate::make("_ctx"));
  TextTemplate::insert(locelEnv, "emitErrors",
                       StrTemplate::make("_emitErrors"));
  TextTemplate::insert(locelEnv, "argsExpansion",
                       StrTemplate::make(argsExpansion));
  os << tmpl.compile(locelEnv);
  return mlir::success();
}

std::string ProductionGen::emitFirstSet(mlir::ArrayAttr attr, StringRef sym) {
  std::string condition;
  llvm::raw_string_ostream cos(condition);
  bool nullable = false;
  llvm::interleave(
      attr.getValue(), cos,
      [&](mlir::Attribute attr) {
        if ((nullable = nullable || isa<mlir::UnitAttr>(attr)))
          return;
        cos << emitLexTerminal(attr, sym, true);
      },
      " || ");
  if (nullable)
    condition = "true";
  return condition;
}

void ProductionGen::emitControlPoint(llvm::raw_ostream &os,
                                     const std::string &label,
                                     const std::string &guard, int ctrlKind) {
  int kind = KindMaskCtrl & ctrlKind;
  bool isNullable = (ctrlKind & NullableCtrl) == NullableCtrl;
  bool isLocal = (ctrlKind & LocalCtrl) == LocalCtrl;
  StringRef knd;
  switch (kind) {
  case SwitchCtrl:
    knd = "CtrlSwitch";
    break;
  case AnyCtrl:
    knd = "CtrlAny";
    break;
  case ZeroOrMoreCtrl:
    knd = "CtrlZeroOrMore";
    break;
  default:
    break;
  }
  auto ctx = contextStack.top();
  StringRef ctrlAction = ctx.controlPoint == "Exit"
                             ? R"(if(ctrl{0} == CtrlNext || ctrl{0} == CtrlExit)
    goto {1};)"
                             : R"(if(ctrl{0} == CtrlNext)
    goto {1};
  else if(ctrl{0} == CtrlExit)
    goto Exit;)";
  os << fmt(R"({0}: {
  auto ctrl{0} = ctrlCheck($ctx, {1}, {2}, {3}, {4}, {5}, $emitErrors);
  {6}
  } // {0}
)",
            label, guard, ctx.guard, knd, isNullable ? "true" : "false",
            isLocal ? "true" : "false",
            fmt(ctrlAction.data(), label, ctx.controlPoint));
}

mlir::LogicalResult ProductionGen::visitSwitch(SwitchOp op,
                                               llvm::raw_ostream &os) {
  // Setup the context:
  os << fmt("  // Switch{0}:\n", switchCount);
  auto guard = fmt("_switch{0}", switchCount);
  auto endLabel = fmt("EndSwitch{0}", switchCount);
  auto emptyLabel = fmt("EndSwitch{0}_0", switchCount++);
  if (contextStack.size() <= 1)
    os << "  {\n";
  contextStack.push(GenContext{endLabel, guard});
  // Emit the switch:
  os << "  $tok = getTok();\n";
  os << "  auto " << guard << " = getGuard($emitErrors);\n";
  for (auto [i, arg] : llvm::enumerate(op.getAlternatives())) {
    // Create the matching condition:
    auto firstSet = dyn_cast<mlir::ArrayAttr>(op.getFirstSets()[i]);
    if (!firstSet)
      return mlir::failure();
    std::string condition = emitFirstSet(firstSet, "$tok");
    if (condition == "true") {
      contextStack.push(GenContext{emptyLabel, guard});
      os << "  $emitErrors = false;\n";
    }
    // Emit the expression:
    os << fmt("  if({0}) {{\n", condition);
    if (mlir::failed(visit(arg.getDefiningOp(), os)))
      return mlir::failure();
    os << fmt("  goto {0};\n", endLabel);
    os << "  }\n";
    if (condition == "true") {
      contextStack.pop();
      emitControlPoint(os, emptyLabel, guard,
                       SwitchCtrl | NullableCtrl | LocalCtrl);
    }
  }
  contextStack.pop();
  // Perform cleanup:
  emitControlPoint(os, endLabel, guard,
                   SwitchCtrl | (op.getNullable() ? NullableCtrl : 0));
  if (contextStack.size() <= 1)
    os << "  }\n";
  return mlir::success();
}

mlir::LogicalResult ProductionGen::visitAny(AnyOp op, llvm::raw_ostream &os) {
  // Setup the context:
  os << fmt("  // Any{0}:\n", anyCount);
  auto guard = fmt("_grdAny{0}", anyCount);
  auto endLabel = fmt("EndAny{0}", anyCount);
  auto count = anyCount++;
  if (contextStack.size() <= 1)
    os << "  {\n";
  contextStack.push(GenContext{endLabel, guard});
  // Emit the any:
  os << "  $tok = getTok();\n";
  os << "  auto " << guard << " = getGuard($emitErrors);\n";
  os << "  $emitErrors = false;\n";
  size_t localCount = {};
  for (auto [i, arg] : llvm::enumerate(op.getAlternatives())) {
    auto endLocal = fmt("EndAny{0}_{1}", count, localCount);
    // Create the matching condition:
    auto firstSet = dyn_cast<mlir::ArrayAttr>(op.getFirstSets()[i]);
    if (!firstSet)
      return mlir::failure();
    std::string condition = emitFirstSet(firstSet, "$tok");
    contextStack.push(GenContext{endLocal, guard});
    os << fmt("  if({0}) {{\n", condition);
    // Emit the expression:
    if (mlir::failed(visit(arg.getDefiningOp(), os)))
      return mlir::failure();
    os << "  }\n";
    contextStack.pop();
    emitControlPoint(os, endLocal, guard,
                     AnyCtrl | (condition == "true" ? NullableCtrl : 0) |
                         LocalCtrl);
    ++localCount;
  }
  contextStack.pop();
  // Perform cleanup:
  emitControlPoint(os, endLabel, guard,
                   AnyCtrl | (op.getNullable() ? NullableCtrl : 0));
  if (contextStack.size() <= 1)
    os << "  }\n";
  return mlir::success();
}

mlir::LogicalResult ProductionGen::visitSequence(SequenceOp op,
                                                 llvm::raw_ostream &os) {
  for (auto arg : op.getAlternatives())
    if (mlir::failed(visit(arg.getDefiningOp(), os)))
      return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult ProductionGen::visitZeroOrMore(ZeroOrMoreOp op,
                                                   llvm::raw_ostream &os) {
  // Setup the context:
  os << fmt("  // ZeroOrMore{0}:\n", zomCount);
  auto guard = fmt("_grdZOM{0}", zomCount);
  auto endLabel = fmt("ZeroOrMore{0}", zomCount++);
  // Emit the zero or more:
  if (contextStack.size() <= 1)
    os << "  {\n";
  contextStack.push(GenContext{endLabel, guard});
  os << "  auto " << guard << " = getGuard($emitErrors);\n";
  os << "  $emitErrors = false;\n";
  os << "  while($lexer.isValid()) {\n";
  os << "  auto _bLocZOM = getLoc();\n";
  os << "  $tok = getTok();\n";
  // Obtain the first set:
  auto firstSet = op.getFirstSetAttr();
  if (!firstSet)
    return mlir::failure();
  std::string condition =
      op.getNullable() ? "" : emitFirstSet(firstSet, "$tok");
  // Emit the expression:
  if (!condition.empty())
    os << fmt(" if ({0}) {{\n", condition);
  if (mlir::failed(visit(op.getExpr().getDefiningOp(), os)))
    return mlir::failure();
  if (!condition.empty())
    os << "  }\n";
  os << fmt(R"(  if (!{0}.isSuccess() || _bLocZOM == getLoc())
    break;
  {0}.update(getGuard());
)",
            guard);
  os << "  }\n";
  contextStack.pop();
  // Perform cleanup:
  emitControlPoint(os, endLabel, guard,
                   ZeroOrMoreCtrl | (op.getNullable() ? NullableCtrl : 0));
  if (contextStack.size() <= 1)
    os << "  }\n";
  return mlir::success();
}

std::string ProductionGen::emitLexTerminal(mlir::Attribute rawAttr,
                                           StringRef sym, bool testEq) {
  auto attr = dyn_cast<LexTerminalAttr>(rawAttr);
  if (!attr)
    return "";
  StringRef litId = attr.getIdentifier().getValue();
  StringRef test = testEq ? "" : "!";
  if (attr.getKind() == LexTerminalKind::Token)
    return fmt("{1}$lexer.isTok({0}, $lexer.{2})", sym, test, litId);
  else if (attr.getKind() == LexTerminalKind::Class)
    return fmt("{1}$lexer.is{2}({0})", sym, test, litId);
  else if (attr.getKind() == LexTerminalKind::Unspecified)
    return fmt("{1}$lexer.isTok({0}, $lexer.{2}, \"{3}\")", sym, test, litId,
               attr.getAlias());
  else if (attr.getKind() == LexTerminalKind::Dynamic)
    return fmt("{1}isDynTok({0}, ::xblang::TypeInfo::get<{2}>())", sym, test,
               attr.getAlias());
  else if (attr.getKind() == LexTerminalKind::Any)
    return testEq ? "true" : "false";
  llvm_unreachable("invalid literal kind");
}

mlir::LogicalResult ProductionGen::visitTerminal(TerminalOp op,
                                                 llvm::raw_ostream &os,
                                                 StringRef name) {
  auto literal = dyn_cast<LexTerminalAttr>(op.getTerminal());
  if (!literal)
    return op.emitError("expected a `lex_literal` attribute");
  if (literal.getKind() == LexTerminalKind::Dynamic) {
    if (name.empty())
      os << "  {\n";
    StringRef code = R"(  auto _tok = getTok();
  auto combinator$cid = getCombinator(_tok, ::xblang::TypeInfo::get<$DynKind>());
  if (!combinator$cid) {
    $diag = emitError(_tok.getLoc(), "expected a `$DynKind` combinator");
    $guard = error();
    goto $controlPoint;
  }
  ::xblang::syntax::combinator_result<$DynKind> $Result;
  auto _srcState$cid = dynParse(combinator$cid, $Result, $emitErrors);
  if($Result.isAnError()) {
    $guard =  error();
    goto $controlPoint;
  }
  setState(_srcState$cid);
  if (getTok().isInvalid()) {
    $guard =  error();
    goto $controlPoint;
  }
  $guard = success();
)";
    auto tmpl = TemplateEngine::make(code);
    tmpl.insert("DynKind", StrTemplate::make(literal.getAlias()));
    tmpl.insert("guard", StrTemplate::make(contextStack.top().guard));
    tmpl.insert("controlPoint",
                StrTemplate::make(contextStack.top().controlPoint));
    tmpl.insert("Result", StrTemplate::make(name.empty() ? "_pc" : name));
    tmpl.insert("cid", StrTemplate::make(fmt("_{0}", combinatorCount++)));
    os << tmpl.compile();
    if (name.empty())
      os << "  }\n";
    return mlir::success();
  }
  if (name.empty())
    os << "  {\n";
  StringRef tok = name.empty() ? "tok" : name;
  os << fmt("  auto {0} = getTok();\n", tok);
  os << fmt("  if({0}) {{\n", emitLexTerminal(literal, tok, false));
  os << fmt("  $diag = emitError({0}.getLoc(), \"expected a `{1}"
            "` token\");\n",
            tok, literal.getIdentifier().getValue());
  os << fmt("  {0} =  error(); goto {1}; }\n", contextStack.top().guard,
            contextStack.top().controlPoint);
  os << "  consume();\n";
  os << fmt("  {0} =  success();\n", contextStack.top().guard);
  if (name.empty())
    os << "  }\n";
  return mlir::success();
}

mlir::LogicalResult ProductionGen::visitNonTerminal(NonTerminalOp op,
                                                    llvm::raw_ostream &os,
                                                    StringRef name) {
  if (op.getNonTerminal() == "_$dyn") {
    if (name.empty())
      os << "  {\n";
    StringRef code = R"(  auto combinator$cid =
    getCombinator("", ::xblang::TypeInfo::get<$DynKind>());
  if (!combinator$cid) {
    $diag = emitError(_tok.getLoc(), "expected a `$DynKind` combinator");
    $guard = error();
    goto $controlPoint;
  }
  ::xblang::syntax::combinator_result<$DynKind> $Result;
  auto _srcState$cid = dynParse(combinator$cid, $Result, $emitErrors);
  if($Result.isAnError()) {
    $guard =  error();
    goto $controlPoint;
  }
  setState(_srcState$cid);
  if (getTok().isInvalid()) {
    $guard = error();
    goto $controlPoint;
  }
  $guard = success();
)";
    auto tmpl = TemplateEngine::make(code);
    tmpl.insert("DynKind", StrTemplate::make(*op.getDynamic()));
    tmpl.insert("guard", StrTemplate::make(contextStack.top().guard));
    tmpl.insert("controlPoint",
                StrTemplate::make(contextStack.top().controlPoint));
    tmpl.insert("Result", StrTemplate::make(name.empty() ? "_pc" : name));
    tmpl.insert("cid", StrTemplate::make(fmt("_{0}", combinatorCount++)));
    os << tmpl.compile();
    if (name.empty())
      os << "  }\n";
    return mlir::success();
  }
  if (name.empty())
    os << "  {\n";
  StringRef nt = name.empty() ? "_nt" : name;
  os << fmt("  auto {0} = parse{1}($emitErrors);\n", nt,
            llvm::convertToCamelFromSnakeCase(op.getNonTerminal(), true));
  os << fmt("  if({0}.isAnError()) {{\n", nt, contextStack.top().controlPoint);
  os << fmt("  {0} =  error(); goto {1}; }\n", contextStack.top().guard,
            contextStack.top().controlPoint);
  if (name.empty())
    os << "  }\n";
  os << fmt("  {0} =  success();\n", contextStack.top().guard);
  return mlir::success();
}

mlir::LogicalResult ProductionGen::visitMetadata(MetadataOp op,
                                                 llvm::raw_ostream &os) {
  auto name = op.getName();
  auto action = op.getCodeActionAttr();
  if (action && !action.getPreAction().empty())
    os << action.getPreAction();
  if (mlir::failed(visit(op.getExpr().getDefiningOp(), os, name ? *name : "")))
    return mlir::failure();
  if (action && !action.getPostAction().empty()) {
    TemplateEngine tmpl = TemplateEngine::make(action.getPostAction());
    tmpl.insert("assertError", TemplateEngine::make(R"({
    $guard = error();
    goto $controlPoint;
})"));
    tmpl.insert("guard", StrTemplate::make(contextStack.top().guard));
    tmpl.insert("controlPoint",
                StrTemplate::make(contextStack.top().controlPoint));
    os << tmpl.compile(tmpl.getEnvironment());
  }
  return mlir::success();
}

mlir::LogicalResult ProductionGen::visitEmptyString(EmptyStringOp op,
                                                    llvm::raw_ostream &os) {
  os << fmt("  {0} =  success();\n", contextStack.top().guard);
  return mlir::success();
}

mlir::LogicalResult ProductionGen::visit(mlir::Operation *op,
                                         llvm::raw_ostream &os,
                                         StringRef name) {
  assert(op && "invalid null operation");
  if (auto swOp = dyn_cast<SwitchOp>(op))
    return visitSwitch(swOp, os);
  else if (auto seqOp = dyn_cast<SequenceOp>(op))
    return visitSequence(seqOp, os);
  else if (auto anyOp = dyn_cast<AnyOp>(op))
    return visitAny(anyOp, os);
  else if (auto zomOp = dyn_cast<ZeroOrMoreOp>(op))
    return visitZeroOrMore(zomOp, os);
  else if (auto terminalOp = dyn_cast<TerminalOp>(op))
    return visitTerminal(terminalOp, os, name);
  else if (auto ntOp = dyn_cast<EmptyStringOp>(op))
    return visitEmptyString(ntOp, os);
  else if (auto ntOp = dyn_cast<NonTerminalOp>(op))
    return visitNonTerminal(ntOp, os, name);
  else if (auto ntOp = dyn_cast<MetadataOp>(op))
    return visitMetadata(ntOp, os);
  return mlir::failure();
}

//===----------------------------------------------------------------------===//
// Generate the parser MLIR module.
//===----------------------------------------------------------------------===//
static mlir::GenRegistration genParserDecls(
    "gen-parser-decls", "Generate parser declarations",
    +[](const llvm::RecordKeeper &records, llvm::raw_ostream &os) {
      return mlir::failed(Generator(records).emitDecl(os));
    });

static mlir::GenRegistration genParserDefs(
    "gen-parser-defs", "Generate parser definitions",
    +[](const llvm::RecordKeeper &records, llvm::raw_ostream &os) {
      return mlir::failed(Generator(records).emitDef(os));
    });

static mlir::GenRegistration dumpLex(
    "dump-parser", "Dump the parser module",
    +[](const llvm::RecordKeeper &records, llvm::raw_ostream &os) {
      return mlir::failed(Generator(records).dump(os));
    });
