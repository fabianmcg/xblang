//===- LexerGen.cpp - Lexer Generator ----------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the TableGen lexer generator.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/TableGen/GenInfo.h"
#include "xblang/Support/Format.h"
#include "xblang/Syntax/IR/SyntaxDialect.h"
#include "xblang/Syntax/LexGen/SDTranslation.h"
#include "xblang/Syntax/Transforms/Passes.h"
#include "xblang/TableGen/Lexer.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

#include "TemplateEngine.h"

using namespace xblang;
using namespace xblang::tablegen;
using namespace xblang::syntaxgen;
using namespace xblang::syntaxgen::lex;

namespace {
llvm::cl::OptionCategory genCat("Options for lexer generators");
}

typedef enum {
  CodeGenStage,
  ProcessSyntaxStage,
  LexToDFAStage,
  MinimizeDFAStage
} PipelineStage;

static llvm::cl::opt<PipelineStage> pipelineStage(
    llvm::cl::desc("Lexer gen final stage:"), llvm::cl::cat(genCat),
    "gen-lexer-opt-stage",
    llvm::cl::values(clEnumValN(CodeGenStage, "codegen", "Initial code gen"),
                     clEnumValN(ProcessSyntaxStage, "lex",
                                "Convert syntax to a lexing rules"),
                     clEnumValN(LexToDFAStage, "dfa", "Convert to DFA"),
                     clEnumValN(MinimizeDFAStage, "min-dfa",
                                "Minimize the DFA")),
    llvm::cl::init(MinimizeDFAStage));

namespace {
using Environment = TextTemplate::Environment;

//===----------------------------------------------------------------------===//
// DFA generator
//===----------------------------------------------------------------------===//
class DFAGen {
public:
  /// Generates the MLIR DFA module.
  static std::optional<DFAGen> genModule(Lexer lexer, DFA automata,
                                         mlir::OpBuilder builder,
                                         xblang::SourceManager &srcMgr);

  /// Emits the lexer declaration.
  mlir::LogicalResult emitDecl(llvm::raw_ostream &os, const Environment &env);

  /// Emits the lexer definition.
  mlir::LogicalResult emitDef(llvm::raw_ostream &os, const Environment &env);

  /// Emits a state definition.
  mlir::LogicalResult emitState(LexStateOp state, llvm::raw_ostream &os,
                                StringRef indent, const Environment &env);

  /// Emits a terminal definition.
  std::string emitTerminal(mlir::Value terminal);

private:
  DFAGen(Lexer lexer, DFA automata);
  /// Parent lexer.
  Lexer lexer;
  /// DFA tablegen class
  DFA automata;
  /// DFA module
  DFAOp dfaModule;
  /// Rule LUT
  llvm::StringMap<Rule> rules;
};

//===----------------------------------------------------------------------===//
// Lexer generator
//===----------------------------------------------------------------------===//
class LexerGen {
public:
  /// Generates the MLIR lexer module.
  static std::optional<LexerGen> genModule(Lexer lexer, mlir::OpBuilder builder,
                                           xblang::SourceManager &srcMgr);

  /// Emits the lexer declaration.
  mlir::LogicalResult emitDecl(llvm::raw_ostream &os);

  /// Emits the lexer definition.
  mlir::LogicalResult emitDef(llvm::raw_ostream &os);

private:
  LexerGen(Lexer lexer);
  /// Lexer being processed.
  Lexer lexer;
  /// Automatons
  SmallVector<DFAGen> automatons;
  /// Lexer module.
  LexerOp lexerModule;
  /// Lexer decl header.
  static llvm::StringRef lexerDecl;
  /// Lexer definition.
  static llvm::StringRef lexerDef;
};

//===----------------------------------------------------------------------===//
// Global generator
//===----------------------------------------------------------------------===//
class Generator {
public:
  Generator(const llvm::RecordKeeper &records);

  /// Emits the lexer declarations.
  mlir::LogicalResult emitDecl(llvm::raw_ostream &os);

  /// Emits the lexer definitions.
  mlir::LogicalResult emitDef(llvm::raw_ostream &os);

  /// Dump the lexers.
  mlir::LogicalResult dumpLexer(llvm::raw_ostream &os);

private:
  /// Initializes the generator.
  void init(const llvm::RecordKeeper &records);

  /// Run the MLIR pass pipeline
  mlir::LogicalResult runPipeline(PipelineStage stage = MinimizeDFAStage);

  /// MLIR context.
  mlir::MLIRContext context;
  /// Source manager.
  xblang::SourceManager sourceManager;
  /// Lexers being generated.
  SmallVector<LexerGen> lexers;
  /// Module containing all lexers.
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
  for (llvm::Record *v : records.getAllDerivedDefinitions("Lexer")) {
    if (auto lex = Lexer::castOrNull(v)) {
      auto lexer = LexerGen::genModule(*lex, builder, sourceManager);
      if (!lexer) {
        llvm::PrintFatalError(v, "failed to process the lexer");
        return;
      }
      lexers.push_back(std::move(*lexer));
    }
  }
  this->module = std::move(module);
}

mlir::LogicalResult Generator::runPipeline(PipelineStage stage) {
  if (!module)
    return mlir::failure();
  mlir::PassManager pm(&context);
  mlir::OpPassManager &dfaPM = pm.nest<LexerOp>().nest<DFAOp>();
  if (stage >= ProcessSyntaxStage)
    dfaPM.addPass(createProcessSyntax(ProcessSyntaxOptions{true}));
  if (stage >= LexToDFAStage)
    dfaPM.addPass(createLexToDFA());
  if (stage >= MinimizeDFAStage)
    dfaPM.addPass(createMinimizeDFA());
  return pm.run(*module);
}

mlir::LogicalResult Generator::dumpLexer(llvm::raw_ostream &os) {
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
  for (auto &lex : lexers)
    if (mlir::failed(lex.emitDecl(os)))
      return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult Generator::emitDef(llvm::raw_ostream &os) {
  if (!module)
    return mlir::failure();
  if (mlir::failed(runPipeline(pipelineStage)))
    return mlir::failure();
  for (auto &lex : lexers)
    if (mlir::failed(lex.emitDef(os)))
      return mlir::failure();
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Lexer generator
//===----------------------------------------------------------------------===//

LexerGen::LexerGen(Lexer lexer) : lexer(lexer) {}

std::optional<LexerGen> LexerGen::genModule(Lexer lexer,
                                            mlir::OpBuilder builder,
                                            xblang::SourceManager &srcMgr) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  LexerGen lexGen(lexer);
  // Create the lexer module.
  auto lexModule = builder.create<LexerOp>(
      builder.getUnknownLoc(), builder.getStringAttr(lexer.getIdentifier()),
      nullptr);
  // Insert the block of the module and set the builder.
  builder.setInsertionPointToEnd(&lexModule.getBodyRegion().emplaceBlock());
  lexGen.lexerModule = lexModule;
  for (auto dfa : lexer.getDFAutomatons()) {
    auto dfaGen = DFAGen::genModule(lexer, dfa, builder, srcMgr);
    if (!dfaGen)
      llvm::PrintFatalError(&dfa.getDef(), "failed to process the DFA");
    lexGen.automatons.push_back(std::move(*dfaGen));
  }
  return std::move(lexGen);
}

llvm::StringRef LexerGen::lexerDecl = R"(
class $LexerName : public ::xblang::syntax::LexerMixin<$LexerName> {
public:
  template <typename>
  friend class ::xblang::syntax::LexerMixin;
  using base = ::xblang::syntax::LexerMixin<$LexerName>;
  using Token = typename base::LexerToken;
  using base::LexerMixin;

  /// Tokens.
  typedef enum {
    Invalid = Token::Invalid,
    EndOfFile = Token::EndOfFile,$Tokens
  } TokenID;

  /// Returns an int as a token ID.
  static inline constexpr TokenID getToken(int value) {
    return static_cast<TokenID>(value);
  }
  /// Returns whether the token matches the expected token.
  static bool isTok(const Token& tok, TokenID expected) {
    return tok.getTok() == expected;
  }
  static bool isTok(const Token& tok, TokenID expected, llvm::StringRef spelling) {
    return tok.getTok() == expected && tok.getSpelling() == spelling;
  }

  /// Converts a token ID to a string representation.
  static llvm::StringRef toString(TokenID value);$TokenClasses

//===----------------------------------------------------------------------===//
// DFA declarations.
//===----------------------------------------------------------------------===//
${LexerDFAs}

//===----------------------------------------------------------------------===//
// Extra class declarations.
//===----------------------------------------------------------------------===//
${ExtraClassDeclarations}

private:
  /// Registers the lexer keywords.
  void registerKeywords();
};
)";

mlir::LogicalResult LexerGen::emitDecl(llvm::raw_ostream &os) {
  TemplateEngine header = TemplateEngine::make(lexerDecl);
  header.insert("LexerName", StrTemplate::make(lexer.getIdentifier()));
  header.insert("ExtraClassDeclarations",
                TemplateEngine::make(lexer.getExtraClassDeclarations()));
  // Generate the tokens
  auto toks = lexer.getTokens();
  std::string enumToks;
  for (auto tok : toks)
    enumToks += fmt("\n    {0},", tok.getName());
  header.insert("Tokens", StrTemplate::make(enumToks));
  // Generate the token classes
  auto tokClasses = lexer.getTokenClasses();
  enumToks = "";
  for (auto tok : tokClasses) {
    auto name = llvm::convertToCamelFromSnakeCase(tok.getName(), true);
    enumToks += "\n  /// Token class: " + tok.getName().str();
    enumToks += fmt("\n  static bool is{0}(TokenID tok);", name);
    enumToks += fmt("\n  static bool is{0}(Token tok) {{\n", name);
    enumToks += fmt("    return is{0}(tok.getTok());\n  }", name);
  }
  header.insert("TokenClasses", StrTemplate::make(enumToks));
  // Generate the DFAs
  std::string dfas;
  llvm::raw_string_ostream sos(dfas);
  for (auto &dfa : automatons) {
    if (mlir::failed(dfa.emitDecl(sos, header.getEnvironment())))
      return mlir::failure();
  }
  header.insert("LexerDFAs", StrTemplate::make(dfas));
  llvm::SmallVector<StringRef, 2> namespaces;
  llvm::SplitString(lexer.getCppNamespace(), namespaces, "::");
  for (StringRef ns : namespaces)
    os << "namespace " << ns << " {\n";
  os << header.compile(header.getEnvironment());
  for (StringRef ns : llvm::reverse(namespaces))
    os << "} // namespace " << ns << "\n";
  return mlir::success();
}

llvm::StringRef LexerGen::lexerDef = R"(
//===----------------------------------------------------------------------===//
// Lexer $LexerName.
//===----------------------------------------------------------------------===//
llvm::StringRef $cppNamespace::$LexerName::toString(TokenID value) {
  switch(value) {
   case Invalid:
     return "Invalid";
   case EndOfFile:
     return "EndOfFile";$TokenCases
  }
  return "";
}
$TokenClasses

void $cppNamespace::$LexerName::registerKeywords() {
$RegisterKeywords
}
${ExtraClassDefinitions}
${LexerDFAs}
)";

mlir::LogicalResult LexerGen::emitDef(llvm::raw_ostream &os) {
  TemplateEngine tmpl = TemplateEngine::make(lexerDef);
  tmpl.insert("LexerName", StrTemplate::make(lexer.getIdentifier()));
  tmpl.insert("ExtraClassDefinitions",
              TemplateEngine::make(lexer.getExtraClassDefinitions()));
  tmpl.insert("cppNamespace", StrTemplate::make(lexer.getCppNamespace()));
  // Generate the tokens
  auto toks = lexer.getTokens();
  std::string enumToks, keywords;
  for (auto tok : toks) {
    enumToks += fmt("\n    case {0}:", tok.getName());
    enumToks += fmt("\n      return \"{0}\";", tok.getName());
    if (auto kw = Keyword::castOrNull(&tok.getDef()))
      keywords +=
          fmt("  addKeyword({0}, \"{1}\");\n", kw->getName(), kw->getKeyword());
  }
  tmpl.insert("TokenCases", StrTemplate::make(enumToks));
  tmpl.insert("RegisterKeywords", StrTemplate::make(keywords));
  // Generate the token classes
  auto tokClasses = lexer.getTokenClasses();
  enumToks = "";
  llvm::StringRef tokTmpl = R"(
bool $cppNamespace::$LexerName::is$ClassName(TokenID tok) {
  switch(tok) {$Cases
  default:
    return false;
  }
}
)";
  for (auto tokClass : tokClasses) {
    auto tmpTmpl = TemplateEngine::make(tokTmpl);
    tmpTmpl.insert("ClassName",
                   StrTemplate::make(llvm::convertToCamelFromSnakeCase(
                       tokClass.getName(), true)));
    std::string cases;
    for (auto tok : tokClass.getTokens())
      cases += fmt("\n  case {0}:", tok.getName());
    if (!cases.empty())
      cases += "\n    return true;";
    tmpTmpl.insert("Cases", StrTemplate::make(cases));
    enumToks += tmpTmpl.compile(tmpl.getEnvironment());
  }
  tmpl.insert("TokenClasses", StrTemplate::make(enumToks));
  // Generate the DFAs
  std::string dfas;
  llvm::raw_string_ostream sos(dfas);
  for (auto &dfa : automatons) {
    if (mlir::failed(dfa.emitDef(sos, tmpl.getEnvironment())))
      return mlir::failure();
  }
  tmpl.insert("LexerDFAs", StrTemplate::make(dfas));
  os << tmpl.compile(tmpl.getEnvironment());
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// DFA generator
//===----------------------------------------------------------------------===//

DFAGen::DFAGen(Lexer lexer, DFA automata) : lexer(lexer), automata(automata) {}

std::optional<DFAGen> DFAGen::genModule(Lexer lexer, DFA automata,
                                        mlir::OpBuilder builder,
                                        xblang::SourceManager &srcMgr) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  DFAGen dfaGen(lexer, automata);
  // Create the dfa module.
  auto dfaModule = builder.create<DFAOp>(
      builder.getUnknownLoc(), builder.getStringAttr(automata.getIdentifier()),
      nullptr);
  // Insert the block of the module and set the builder.
  builder.setInsertionPointToEnd(&dfaModule.getBodyRegion().emplaceBlock());
  dfaGen.dfaModule = dfaModule;
  for (auto def : automata.getDefinitions())
    SDTranslator(*srcMgr.createSource(def.getRule(), def.getIdentifier()))
        .parseDefinition(def.getIdentifier(), *dfaModule.getBody(0));
  for (auto rule : automata.getRules()) {
    StringRef id = rule.getAsToken().getName();
    SDTranslator(*srcMgr.createSource(rule.getRule(), id))
        .parseRule(id, *dfaModule.getBody(0));
    dfaGen.rules.insert_or_assign(id, rule);
  }
  return std::move(dfaGen);
}

mlir::LogicalResult DFAGen::emitDecl(llvm::raw_ostream &os,
                                     const Environment &env) {
  TemplateEngine decl = TemplateEngine::make(std::string(R"(
//===----------------------------------------------------------------------===//
// DFA ${DFAName}
//===----------------------------------------------------------------------===//
  TokenID lex${DFAName}(::xblang::SourceState &state,
                        ::xblang::SourceLocation &beginLoc,
                        llvm::StringRef &spelling) const;

  inline Token lex${DFAName}(::xblang::SourceState &state) const {
    SourceLocation loc;
    llvm::StringRef spelling;
    auto tok = lex${DFAName}(state, loc, spelling);
    return Token(tok, spelling, loc);
  }
)"));
  decl.insert("DFAName", StrTemplate::make(automata.getIdentifier()));
  os << decl.compile(env);
  return mlir::success();
}

mlir::LogicalResult DFAGen::emitDef(llvm::raw_ostream &os,
                                    const Environment &env) {
  TemplateEngine def = TemplateEngine::make(std::string(R"(
//===----------------------------------------------------------------------===//
// DFA $LexerName::${DFAName}
//===----------------------------------------------------------------------===//
$cppNamespace::$LexerName::TokenID $cppNamespace::$LexerName::lex${DFAName}(
          ::xblang::SourceState &state,
          ::xblang::SourceLocation &beginLoc,
          llvm::StringRef &spelling) const {
$DFADef
  return Invalid;
}
)"));
  def.insert("DFAName", StrTemplate::make(automata.getIdentifier()));
  bool inLoop = automata.getLoop();
  std::string dfa, indent = inLoop ? "    " : "  ";
  llvm::raw_string_ostream osDfa(dfa);
  if (inLoop)
    osDfa << "  while(true) {\n";
  if (automata.getIgnoreWhitespace()) {
    osDfa << R"(    if (isspace(*state))
      while (isspace(*state.advance()))
        ;
    beginLoc = state.getLoc();
    if (!*state)
      return EndOfFile;
)";
  }
  mlir::WalkResult walkResult =
      dfaModule.walk([&](LexStateOp op) -> mlir::WalkResult {
        if (mlir::failed(emitState(op, osDfa, indent, env)))
          return mlir::WalkResult::interrupt();
        return mlir::WalkResult::skip();
      });
  if (walkResult.wasInterrupted())
    return mlir::failure();
  if (inLoop)
    osDfa << "    break;\n"
          << "  }\n";
  def.insert("DFADef", StrTemplate::make(dfa));
  os << def.compile(env);
  return mlir::success();
}

mlir::LogicalResult DFAGen::emitState(LexStateOp state, llvm::raw_ostream &os,
                                      StringRef indent,
                                      const Environment &env) {
  os << fmt("{0}: {{\n", state.getName());
  // Emit final states
  if (state.getFinalState()) {
    auto id = state.getId();
    if (!id)
      return mlir::failure();
    auto it = rules.find(*id);
    if (it == rules.end())
      return mlir::failure();
    Rule rule = it->second;
    os << fmt("/*Rule: {0}*/", *id);
    os << indent << "if (true) {";
    auto engine = TemplateEngine::make(rule.getAction());
    engine.insert("token", StrTemplate::make(rule.getAsToken().getName()));
    os << engine.compile(env);
    os << indent << "}\n";
    os << indent << "return Invalid;\n";
    os << "}\n";
    return mlir::success();
  }
  // Emit states with transitions
  mlir::WalkResult walkResult =
      state.walk([&](LexTransitionOp top) -> mlir::WalkResult {
        std::string terminal = emitTerminal(top.getTerminal());
        if (terminal.empty())
          return mlir::WalkResult::interrupt();
        os << indent << fmt("if({0}) {{\n", terminal);
        if (terminal != "true")
          os << indent << "  state.advance();\n";
        os << indent << fmt("  goto {0};\n", top.getNextState());
        os << indent << " }\n";
        return mlir::WalkResult::skip();
      });
  if (walkResult.wasInterrupted())
    return mlir::failure();
  os << indent << "return Invalid;\n";
  os << "}\n";
  return mlir::success();
}

std::string DFAGen::emitTerminal(mlir::Value terminal) {
  mlir::Operation *dop = terminal.getDefiningOp();
  if (!dop) {
    llvm::PrintFatalError("terminal has no definition");
    return "";
  }
  std::string condition;
  llvm::raw_string_ostream os(condition);
  if (auto op = llvm::dyn_cast<TerminalOp>(dop)) {
    if (auto literal = llvm::dyn_cast<LiteralAttr>(op.getTerminal()))
      os << fmt("*state == {0} /*{1}*/", literal.getLiteral(),
                utfToString(literal.getLiteral()));
    else if (auto attr = llvm::dyn_cast<CharClassAttr>(op.getTerminal())) {
      auto fn = [&](const CharRange &rng) {
        if (rng.isChar())
          os << fmt("*state == {0} /*{1}*/", rng.getLower(),
                    utfToString(rng.getLower()));
        else
          os << fmt("(/*{1}*/ ({0} <= *state) && (*state <= {2}) /*{3}*/)",
                    rng.getLower(), utfToString(rng.getLower()), rng.getUpper(),
                    utfToString(rng.getUpper()));
      };
      llvm::interleave(attr.getCharClass().getRanges(), os, fn, " || ");
    } else
      llvm::PrintFatalError(
          "invalid terminal attribute, it has to be a literal or a CharClass");
  } else if (auto op = llvm::dyn_cast<EmptyStringOp>(dop))
    os << "true";
  if (condition.empty())
    llvm::PrintFatalError("invalid terminal");
  return condition;
}

//===----------------------------------------------------------------------===//
// Generate the lexer MLIR module.
//===----------------------------------------------------------------------===//
static mlir::GenRegistration genLexDecls(
    "gen-lexer-decls", "Generate lexer declarations",
    +[](const llvm::RecordKeeper &records, llvm::raw_ostream &os) {
      return mlir::failed(Generator(records).emitDecl(os));
    });

static mlir::GenRegistration genLexDefs(
    "gen-lexer-defs", "Generate lexer definitions",
    +[](const llvm::RecordKeeper &records, llvm::raw_ostream &os) {
      return mlir::failed(Generator(records).emitDef(os));
    });

static mlir::GenRegistration dumpLex(
    "dump-lexer", "Dump the lexer module",
    +[](const llvm::RecordKeeper &records, llvm::raw_ostream &os) {
      return mlir::failed(Generator(records).dumpLexer(os));
    });
