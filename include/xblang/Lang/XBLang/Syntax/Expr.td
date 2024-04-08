#ifndef XBLANG_EXPR_SYNTAX
#define XBLANG_EXPR_SYNTAX

include "xblang/Syntax/SyntaxBase.td"

//===----------------------------------------------------------------------===//
// XB parse a expression
//===----------------------------------------------------------------------===//

def XB_ExprStx : Production<"Expr", "::mlir::Value"> {
  let implement = 0;
  let rule = [{
    BinaryExpr
  }];
}

//===----------------------------------------------------------------------===//
// XB parse a binary expression
//===----------------------------------------------------------------------===//

def XB_BinaryExprStx : Production<"BinaryExpr", "::mlir::Value"> {
  let implement = 0;
  let arguments = (ins "int":$precedence, "syntax::ParseResult<mlir::Value>":$lhs);
  let rule = [{
    TopExpr
  }];
}

//===----------------------------------------------------------------------===//
// XB parse a top expression
//===----------------------------------------------------------------------===//

def XB_TopExprStx : Production<"TopExpr", "::mlir::Value"> {
  let implement = 0;
  let rule = [{
    ( "+"
    | "-"
    | "*"
    | "&"
    | "!"
    | "++"
    | "--"
    )*
    CoreExpr
    (
      "(" @Interleave(Expr, ",") ")" |
      "[" @Interleave(Expr, ",") "]" |
      "++" |
      "--"
    )*
  }];
}

//===----------------------------------------------------------------------===//
// XB parse a core top expression
//===----------------------------------------------------------------------===//

def XB_CoreExprStx : Production<"CoreExpr", "::mlir::Value"> {
  let rule = [{
    ConstExpr {{
      $res = _0.get();
    }} |
    #dyn_kw("xlg::Expr") {{
      $res = _0.get();
    }}  |
    QualifiedIdentifierExpr {{
      $res = _0.get();
    }} |
    "(" Expr ")" {{
      $res = _1;
    }} | {{ Value type = {}; }}
    "[" CommaListExpr:$list "]" (Type:$ty {{ type = ty.get(); }} )? {{
      $res = create<ListExpr>(getLoc(_0),
                              getConceptClass<ListExprCep>(),
                              nullptr,
                              list.get(),
                              type);
    }} |
    "cast" "<" Type:$type ">" "(" Expr:$expr ")" {{
      $res = create<CastExpr>(getLoc(_0),
                              getConceptClass<CastExprCep>(),
                              nullptr,
                              expr.get(),
                              type.get());
    }}
  }];
}

//===----------------------------------------------------------------------===//
// XB parse a constant expression
//===----------------------------------------------------------------------===//

def XB_ConstExprStx : Production<"ConstExpr", "::mlir::Value"> {
  let rule = [{
    IntLit {{
      $res = getIntExpr(_0);
    }} |
    FloatLit {{
      $res = getFloatExpr(_0);
    }} |
    BoolLit {{
      $res = getBoolExpr(_0);
    }} |
    StringLiteral {{
      $res = getStringExpr(_0);
    }} |
    "null" {{
      $res = getNullExpr();
    }}
  }];
}

//===----------------------------------------------------------------------===//
// XB parse a comma list of expressions
//===----------------------------------------------------------------------===//

def XB_CommaListExprStx : Production<"CommaListExpr",
                                     "::mlir::SmallVector<::mlir::Value>"> {
  let rule = [{ {{ ::mlir::SmallVector<::mlir::Value> exprs; }}
    @Interleave(Expr:$expr {{ exprs.push_back(expr.get()); }} , ",")
    {{
      $res = std::move(exprs);
    }}
  }];
}

//===----------------------------------------------------------------------===//
// XB parse a range expression
//===----------------------------------------------------------------------===//

def XB_RangeExprStx : Production<"RangeExpr", "::mlir::Value"> {
  let implement = 0;
  let rule = [{
    ":" ((
        "+"
      | "-"
      | "*"
      | "/"
      | "<<"
      | ">>")?
      Expr ":"
    )? (
      "<"
    | ">"
    | "<="
    | ">="
    | "=="
    | "!="
    )? Expr
  }];
  let arguments = (ins "Value":$expr);
}

//===----------------------------------------------------------------------===//
// XB parse a qualified identifier expression
//===----------------------------------------------------------------------===//

def XB_QualifiedIdentifierExprStx : Production<"QualifiedIdentifierExpr", "::mlir::Value"> {
  let implement = 0;
  let rule = [{
    ("::")? Identifier
  }];
}

#endif // XBLANG_EXPR_SYNTAX
