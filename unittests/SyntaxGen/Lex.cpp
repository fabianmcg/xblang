#ifndef UNITTESTS_SYNTAXGEN_LEX_CPP
#define UNITTESTS_SYNTAXGEN_LEX_CPP

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "xblang/Syntax/IR/SyntaxDialect.h"
#include "xblang/Syntax/LexGen/SDTranslation.h"
#include "llvm/Support/raw_ostream.h"
#include <gtest/gtest.h>

namespace {
TEST(SyntaxGen, SDTLexer) {
  std::string error;
  llvm::raw_string_ostream os(error);
  mlir::MLIRContext context;
  xblang::SourceManager manager(&context, &os);
  manager.registerDiagnosticsHandler();
  xblang::Source *source =
      manager.createSource("lorem _ipsum 23 [* /**/ [a-z\\]] /* 0", "buffer");
  EXPECT_TRUE(!!source);
  xblang::syntaxgen::lex::SDTLexer lexer(manager);
  auto state = source->getState();
  lexer(state);
  EXPECT_EQ(lexer.getTok().getTok(), lexer.Identifier);
  EXPECT_EQ(lexer.consume().getTok(), lexer.Identifier);
  EXPECT_EQ(lexer.consume().getTok(), lexer.Int);
  EXPECT_EQ(lexer.consume().getTok(), lexer.LBracket);
  EXPECT_EQ(lexer.consume().getTok(), lexer.Multiply);
  EXPECT_EQ(lexer.consume().getTok(), lexer.LBracket);
  lexer.setLexChars(true);
  EXPECT_EQ(lexer.consume().getTok(), lexer.Char);
  EXPECT_EQ(lexer.consume().getTok(), lexer.Dash);
  EXPECT_EQ(lexer.consume().getTok(), lexer.Char);
  EXPECT_EQ(lexer.consume().getTok(), lexer.Char);
  EXPECT_EQ(lexer.consume().getTok(), lexer.RBracket);
  lexer.setLexChars(false);
  EXPECT_EQ(lexer.consume().getTok(), lexer.Invalid);
  EXPECT_TRUE(error.find("buffer:1:33: error: the comment was never "
                         "closed\nlorem _ipsum 23 [* /**/ [a-z\\]] /* 0") == 0);
}

TEST(SyntaxGen, SDTranslation) {
  std::string error;
  llvm::raw_string_ostream os(error);
  mlir::MLIRContext context;
  context.loadDialect<xblang::syntaxgen::SyntaxDialect>();
  xblang::SourceManager manager(&context, &os);
  manager.registerDiagnosticsHandler();
  mlir::OwningOpRef<mlir::ModuleOp> uniqueModuleOp =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  auto moduleOp = uniqueModuleOp.get();
  xblang::Source *source = manager.createSource("'0'*", "buffer");
  EXPECT_TRUE(!!source);
  {
    xblang::syntaxgen::lex::SDTranslator translator(*source);
    EXPECT_TRUE(
        translator.parseRule("my_rule", *moduleOp.getBody(0)).isSuccess());
  }
}
} // namespace

#endif
