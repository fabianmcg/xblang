#ifndef PARALLEL_SYNTAX
#define PARALLEL_SYNTAX

include "xblang/Syntax/SyntaxBase.td"
include "xblang/Syntax/CommonLex.td"
include "xblang/Lang/XBLang/Syntax/Syntax.td"


//===----------------------------------------------------------------------===//
// Par parse a statement
//===----------------------------------------------------------------------===//

def Par_CombinatorStmtStx :
    Production<"CombinatorStmt", "::mlir::Operation*"> {
  let rule = [{
    {{
      auto dynScope = pushScope();
      registerCombinator<xlg::Expr>("threadId", invokeExpr);
      registerCombinator<xlg::Expr>("groupId", invokeExpr);
      registerCombinator<xlg::Expr>("groupDim", invokeExpr);
      registerCombinator<xlg::Expr>("gridDim", invokeExpr);
    }}
    "par" Stmt:$stmt {{
      $res = stmt.get();
    }}
  }];
}

def Par_StmtStx : Production<"Stmt", "::mlir::Operation*"> {
  let rule = [{
    Loop
  }];
}

def Par_LoopStx : Production<"Loop", "::mlir::Operation*"> {
  let rule = [{
    "loop" "(" ")"
    #dyn("xlg::Stmt")
  }];
}

def Par_CombinatorExprStx :
    Production<"CombinatorExpr", "::mlir::Value"> {
  let rule = [{
    "par" Expr:$expr {{
      $res = expr.get();
    }}
  }];
}

def Par_ExprStx : Production<"Expr", "::mlir::Value"> {
  let rule = [{
    "threadId" {{
      auto loc = getLoc(_0);
      auto idOp = builder.create<mlir::par::IdOp>(loc,
        ::mlir::par::ParallelHierarchy::m2s);
      $res = builder.create<xbg::ToXLGExpr>(loc,
        getConceptClass<xlg::Expr>(), TypeAttr::get(idOp.getType()), idOp);
    }} |
    "groupId" {{
      auto loc = getLoc(_0);
      auto idOp = builder.create<mlir::par::IdOp>(loc,
        ::mlir::par::ParallelHierarchy::matrix);
      $res = builder.create<xbg::ToXLGExpr>(loc,
        getConceptClass<xlg::Expr>(), TypeAttr::get(idOp.getType()), idOp);
    }} |
    "groupDim"
  | "gridDim"
  }];
}

//===----------------------------------------------------------------------===//
// Par parser
//===----------------------------------------------------------------------===//

def ParSyntax : Parser<"ParParser", XBLangLexer,
    [ParserTrait<"XLGBuilderRef", "::xblang::xlg">]> {
  let macros = [
    ParserMacro<"Interleave", ["expr", "sep"], [{
        expr (sep expr)* | eps
    }]>,
  ];
  let productions = [
    /*par combinators*/
    Par_CombinatorStmtStx,
    Par_CombinatorExprStx,
    /*par syntax*/
    Par_StmtStx,
    Par_LoopStx,
    Par_ExprStx
  ];
  let startSymbol = "Stmt";
  let defaultToken = "Identifier";
  let cppNamespace = "xblang";
  let extraClassDeclaration = [{
    $parserName(XBContext *ctx, ::xblang::xlg::XLGBuilder& builder,
                ::xblang::SourceManager& srcManager, Lexer& lexer);
  }];
}

#endif // XBLANG_SYNTAX
