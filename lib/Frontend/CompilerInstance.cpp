#include "xblang/Frontend/CompilerInstance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "xblang/Basic/ContextDialect.h"
#include "xblang/Dialect/XBLang/IR/Dialect.h"
#include "xblang/Frontend/CompilerInvocation.h"
#include "xblang/Frontend/MLIRPipeline.h"
#include "xblang/Interfaces/Syntax.h"
#include "xblang/Lang/InitAll.h"
#include "xblang/Lang/XBLang/Syntax/Syntax.h"
#include "xblang/Sema/TypeSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SourceMgr.h"

namespace xblang {
CompilerInstance::CompilerInstance(CompilerInvocation &invocation)
    : invocation(invocation), mlirContext(new mlir::MLIRContext()) {
  typeContext = std::unique_ptr<xb::XBLangTypeSystem>(
      new xb::XBLangTypeSystem(*mlirContext));
  auto xbCtxDialect =
      mlirContext->getOrLoadDialect<::xblang::XBContextDialect>();
  mlir::DialectRegistry registry;
  registerXBLang(registry);
#if XBC_INCLUDE_MLIR_DIALECTS == 1
  mlir::registerAllDialects(registry);
#endif
  mlirContext->appendDialectRegistry(registry);
  mlirContext->loadAllAvailableDialects();
  sourceManager =
      std::unique_ptr<SourceManager>(new SourceManager(mlirContext.get()));
  sourceManager->registerDiagnosticsHandler();
  //  sources.setIncludeDirs(this->invocation.getIncludeDirs());
  module = mlir::ModuleOp::create(mlir::UnknownLoc::get(mlirContext.get()));
  xblangContext = &xbCtxDialect->getContext();
}

CompilerInstance::~CompilerInstance() {}

int CompilerInstance::parse() {
  SyntaxContext context(*sourceManager);
  context.getOrRegisterLexer<XBLangLexer>();
  context.getOrRegisterParser<XBLangParser>(xblangContext,
                                            (mlir::Block *)nullptr);
  mlir::DialectInterfaceCollection<xblang::SyntaxDialectInterface> collection(
      &getMLIRContext());
  for (auto &interface : collection)
    interface.populateSyntax(xblangContext, context);
  for (auto file : invocation.getInputFiles()) {
    auto src = sourceManager->getSource(file);
    XBLangParser *parser = context.getParser<XBLangParser>();
    parser->setInsertionPointToEnd(module->getBody());
    assert(parser && "invalid parser");
    if (!parser->parseState(src->getState()).isSuccess())
      return 1;
  }
  return 0;
}

int CompilerInstance::run() {
  int error = 0;
  mlir::DialectRegistry registry;
  for (auto extension : invocation.getExtensions()) {
    auto plugin = mlir::DialectPlugin::load(extension);
    if (!plugin) {
      llvm::errs() << "Failed to load extension plugin from '" << extension
                   << "'. Request ignored.\n";
      return -4;
    };
    plugin.get().registerDialectRegistryCallbacks(registry);
  }
  mlirContext->appendDialectRegistry(registry);
  mlirContext->loadAllAvailableDialects();
  if (invocation.runStage(CompilerInvocation::Parse)) {
    error = parse();
    if (error)
      return error;
    if (invocation.dump() && invocation.isFinalStage(CompilerInvocation::Parse))
      module->print(llvm::errs());
  }
  if (invocation.runStage(CompilerInvocation::Sema)) {
    MLIRPipeline pipeline(*this);
    error = pipeline.run();
    if (error)
      return error;
  }
  return 0;
}

std::unique_ptr<llvm::raw_ostream>
CompilerInstance::getOutput(llvm::Twine filePath) {
  std::error_code code;
  std::string name = filePath.str();
  std::unique_ptr<llvm::raw_ostream> result(
      new llvm::raw_fd_ostream(name, code));
  if (code) {
    llvm::errs() << code.message() << "\n";
    return {};
  }
  return result;
}
} // namespace xblang
