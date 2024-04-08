#ifndef XBLANG_STMT_SYNTAX
#define XBLANG_STMT_SYNTAX

include "xblang/Syntax/SyntaxBase.td"

//===----------------------------------------------------------------------===//
// XB parse a statement
//===----------------------------------------------------------------------===//

def XB_StmtStx : Production<"Stmt", "::mlir::Operation*"> {
  let rule = [{
    CompoundStmt
  | ForStmt
  | IfStmt
  | WhileStmt
  | BreakStmt
  | ContinueStmt
  | ReturnStmt
  | DeclStmt
  | #dyn_kw("xlg::Stmt")
  | Expr ";"
  | ";"
  }];
}

//===----------------------------------------------------------------------===//
// XB parse a return statement
//===----------------------------------------------------------------------===//

def XB_BreakStmtStx : Production<"BreakStmt", "::mlir::Operation*"> {
  let rule = [{
    "break" ";"
    {{
      $res = create<YieldStmt>(getLoc(_0),
                               getConceptClass<YieldStmtCep>(),
                               xb::YieldKind::Break);
    }}
  }];
}

//===----------------------------------------------------------------------===//
// XB parse a compound statement
//===----------------------------------------------------------------------===//

def XB_CompoundStmtStx : Production<"CompoundStmt", "::mlir::Operation*"> {
  let rule = [{ {{ auto block = getInsertionBlock(); }}
    "{"
      Stmt*
    "}" {{
      // Restore the insertion point.
      block.restorePoint();
      auto stmt = create<CompoundStmt>(getLoc(_0),
                                       getConceptClass<CompoundStmtCep>());
      stmt.getBodyRegion().push_back(block.release());
      $res = stmt;
    }}
  }];
}

//===----------------------------------------------------------------------===//
// XB parse a continue statement
//===----------------------------------------------------------------------===//

def XB_ContinueStmtStx : Production<"ContinueStmt", "::mlir::Operation*"> {
  let rule = [{
    "continue" ";"
    {{
      $res = create<YieldStmt>(getLoc(_0),
                               getConceptClass<YieldStmtCep>(),
                               xb::YieldKind::Continue);
    }}
  }];
}

//===----------------------------------------------------------------------===//
// XB parse a declaration statement
//===----------------------------------------------------------------------===//

def XB_DeclStmtStx : Production<"DeclStmt", "::mlir::Value"> {
  let rule = [{
    "let" VarDecl ";" {{
      $res = _1.get();
    }}
  }];
}

//===----------------------------------------------------------------------===//
// XB parse a for statement
//===----------------------------------------------------------------------===//

def XB_ForStmtStx : Production<"ForStmt", "::mlir::Operation*"> {
  let rule = [{
    {{
        InsertionBlock bodyRegion;
        SmallVector<RangeLoopInfo> ranges;
    }}

    "for" "("
      /*parse the ranges*/
      @Interleave({{Value type{}; Token id{};}}
        ( /*parse the iterator*/
          "let" Identifier:$itId (":" Type:$ty {{ type = ty.get(); }})? {{
            id = itId;
          }} |
          Identifier:$itId {{
            id = itId;
          }}
        ) "in" Expr:$expr {{ ranges.push_back({id, type, expr.get()}); }},
        ","
      )
    ")"
    ( /*parse the body*/
      {{ bodyRegion = getInsertionBlock(); }} Stmt
    ) {{
      $res = createRangeForStmt(getLoc(_0), std::move(bodyRegion), ranges);
    }}
  }];
}

//===----------------------------------------------------------------------===//
// XB parse an if statement
//===----------------------------------------------------------------------===//

def XB_IfStmtStx : Production<"IfStmt", "::mlir::Operation*"> {
  let rule = [{ {{ InsertionBlock thenBlock, elseBlock; }}
    "if" "(" Expr:$expr ")"
    /*parse then statement*/ (
      {{ thenBlock = getInsertionBlock(); }}
      Stmt
    ) /*try parsing the else statement*/ (
      "else" (
        {{ elseBlock = getInsertionBlock(); }}
        Stmt
      )
    )?
    {{
      thenBlock.restorePoint();
      elseBlock.abandonPoint();
      auto stmt = create<IfStmt>(getLoc(_0),
                                 getConceptClass<IfStmtCep>(),
                                 expr.get());
      stmt.getThenRegion().push_back(thenBlock.release());
      if (auto block = elseBlock.release())
        stmt.getElseRegion().push_back(block);
      $res = stmt;
    }}
  }];
}

//===----------------------------------------------------------------------===//
// XB parse a return statement
//===----------------------------------------------------------------------===//

def XB_ReturnStmtStx : Production<"ReturnStmt", "::mlir::Operation*"> {
  let rule = [{
    {{ mlir::Value retExpr; }}
    "return" (
      Expr:$expr
      {{ retExpr = expr; }}
    )? ";"
    {{
      $res = create<ReturnStmt>(getLoc(_0),
                                getConceptClass<ReturnStmtCep>(),
                                retExpr);
    }}
  }];
}

//===----------------------------------------------------------------------===//
// XB parse a while statement
//===----------------------------------------------------------------------===//

def XB_WhileStmtStx : Production<"WhileStmt", "::mlir::Operation*"> {
  let rule = [{ {{InsertionBlock block;}}
    "while" "(" Expr:$expr ")"
    (
      {{ block = getInsertionBlock(); }}
      Stmt
    )
    {{
      block.restorePoint();
      auto stmt = create<WhileStmt>(getLoc(_0),
                                    getConceptClass<WhileStmtCep>(),
                                    expr.get());
      stmt.getBodyRegion().push_back(block.release());
      $res = stmt;
    }}
  }];
}

#endif // XBLANG_STMT_SYNTAX
