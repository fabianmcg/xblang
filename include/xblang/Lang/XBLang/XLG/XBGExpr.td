#ifndef XBLANG_XBG_EXPR_OPS
#define XBLANG_XBG_EXPR_OPS

include "xblang/Lang/XBLang/XLG/XBGDialect.td"
include "xblang/XLG/Concepts.td"

include "xblang/Dialect/XBLang/IR/Enums.td"


//===----------------------------------------------------------------------===//
// XBG Expressions
// Notice: all constructs must end in _expr so tablegen can generate them.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// XBG array expression
//===----------------------------------------------------------------------===//

def XBG_ArrayExpr : XBG_Construct<"ArrayExprCep", "array_expr", [ExprCep]> {
  let summary = "XBG array expression";
  let description = [{
    Syntax:
    ```mlir
    %e = xbg.array_expr %array [...]
    ```
  }];
  let assemblyFormat = [{
    custom<Concept>(type($conceptClass)) $base `[` type($base) `]` ` `
   `[` $indexes `:` type($indexes) `]`
    attr-dict
  }];
  let args = (ins ExprCep:$base, Variadic<ExprCep>:$indexes);
}

//===----------------------------------------------------------------------===//
// XBG const expression
//===----------------------------------------------------------------------===//

def XBG_ConstExpr : XBG_Construct<"ConstExprCep", "const_expr", [ConstExprCep]> {
  let summary = "XBG constant expression";
  let description = [{
    Syntax:
    ```mlir
    %e = xbg.const_expr 0 : i32
    ```
  }];
  let assemblyFormat = [{
    custom<Concept>(type($conceptClass)) $expr attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// XBG int expressions
//===----------------------------------------------------------------------===//

def XBG_IntExpr : XBG_PureConstruct<"IntExpr", "int_expr", [XBG_ConstExpr]> {
}

//===----------------------------------------------------------------------===//
// XBG float expressions
//===----------------------------------------------------------------------===//

def XBG_FloatExpr : XBG_PureConstruct<"FloatExpr", "float_expr", [XBG_ConstExpr]> {
}

//===----------------------------------------------------------------------===//
// XBG bool expressions
//===----------------------------------------------------------------------===//

def XBG_BoolExpr : XBG_PureConstruct<"BoolExpr", "bool_expr", [XBG_ConstExpr]> {
}

//===----------------------------------------------------------------------===//
// XBG binary expression
//===----------------------------------------------------------------------===//

def XBG_BinOpExpr : XBG_Construct<"BinOpExprCep", "bin_op_expr", [ExprCep]> {
  let summary = "XBG binary expression";
  let description = [{
    Syntax:
    ```mlir
    %e = xbg.bin_op_expr %lhs, %rhs
    ```
  }];
  let assemblyFormat = [{
    custom<Concept>(type($conceptClass))
    $lhs `:` type($lhs) $op $rhs `:` type($rhs) attr-dict
  }];
  let args = (ins BinaryOperator:$op, ExprCep:$lhs, ExprCep:$rhs);
}

//===----------------------------------------------------------------------===//
// XBG call expression
//===----------------------------------------------------------------------===//

def XBG_CallExpr : XBG_Construct<"CallExprCep", "call_expr", [CallExprCep]> {
  let summary = "XBG call expression";
  let description = [{
    Syntax:
    ```mlir
    %e = xbg.call @sym (...)
    ```
  }];
  let assemblyFormat = [{
    custom<Concept>(type($conceptClass)) $callee`[` type($callee) `]`
    (`(` $arguments^ `)` `:` type($arguments))?
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// XBG cast expressions
//===----------------------------------------------------------------------===//

def XBG_CastExpr : XBG_Construct<"CastExprCep", "cast_expr", [CastExprCep]> {
  let summary = "XBG cast expression";
  let description = [{
    Syntax:
    ```mlir
    %e = xbg.cast_expr %expr : %type
    ```
  }];
  let assemblyFormat = [{
    custom<Concept>(type($conceptClass)) $expr `[`type($expr)`]`
    (`:` $dstType^ `[`type($dstType)`]` )?
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// XBG FromXLG expression
//===----------------------------------------------------------------------===//

def XBG_FromXLGExpr :  XBG_Construct<"FromXLGExprCep", "from_xlg_expr",
    [ExprCep]> {
  let summary = "Transforms an XLG value to MLIR.";
  let description = [{
    This operation transforms an XLG value to an MLIR value.
    Syntax:
    ```mlir
    %1 = xbg.from_xlg %0 : xlg.class_type<#xbg.expr> : i32
    ```
  }];
  let args = (ins XLG_ConceptType:$expr);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
     custom<Concept>(type($expr)) $expr `:` type($result) attr-dict
  }];
  let extraClassDeclaration = [{
    xblang::xlg::ConceptType getConceptClass() {
      return getExpr().getType();
    }
  }];
}

//===----------------------------------------------------------------------===//
// XBG list expression
//===----------------------------------------------------------------------===//

def XBG_ListExpr : XBG_Construct<"ListExprCep", "list_expr", [ExprCep],
    [AttrSizedOperandSegments]> {
  let summary = "XBG list expression";
  let description = [{
    Syntax:
    ```mlir
    %e = xbg.list_expr [...]
    ```
  }];
  let args = (ins Variadic<ExprCep>:$elements, Optional<TypeCep>:$elementType);
  let assemblyFormat = [{
    custom<Concept>(type($conceptClass))
    `[` $elements `:` type($elements) `]`
    ($elementType^ `:` type($elementType))?
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// XBG load expression
//===----------------------------------------------------------------------===//

def XBG_LoadExpr :
    XBG_Construct<"LoadExprCep", "load_expr", [ExprCep]> {
  let summary = "XBG load expression";
  let description = [{
    Syntax:
    ```mlir
    %e = xbg.load_expr %value_to_Load
    ```
  }];
  let args = (ins ExprCep:$address);
  let assemblyFormat = [{
    custom<Concept>(type($conceptClass)) $address `[` type($address) `]` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// XBG member reference expression
//===----------------------------------------------------------------------===//

def XBG_MemberRefExpr :
    XBG_Construct<"MemberRefExprCep", "member_ref_expr", [SymbolRefExprCep]> {
  let summary = "XBG member reference expression";
  let description = [{
    Syntax:
    ```mlir
    %e = xbg.member_ref_expr %base, %referenced_member
    ```
  }];
  let args = (ins ExprCep:$base);
  let assemblyFormat = [{
    custom<Concept>(type($conceptClass))
    `base` $base `[` type($base) `]` `member` $sym_name attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// XBG range expression
//===----------------------------------------------------------------------===//

def XBG_RangeExpr : XBG_Construct<"RangeExprCep", "range_expr", [ExprCep]> {
  let summary = "XBG range expression";
  let description = [{
    Syntax:
    ```mlir
    %e = xbg.range_expr %expr : %type
    ```
  }];
  let args = (ins OptionalAttr<BinaryOperator>:$comparator,
                       ExprCep:$begin,
                       ExprCep:$end,
                       OptionalAttr<BinaryOperator>:$stepOp,
                       Optional<ExprCep>:$step);
  let assemblyFormat = [{
    custom<Concept>(type($conceptClass))
    $begin `[` type($begin) `]` ($comparator^)? $end `[` type($end) `]`
    (`step` ($stepOp^)? $step^ `[` type($step) `]`)? attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// XBG reference expressions
//===----------------------------------------------------------------------===//

def XBG_RefExpr : XBG_Construct<"RefExprCep", "ref_expr", [SymbolRefExprCep]> {
  let summary = "XBG reference expression";
  let description = [{
    Syntax:
    ```mlir
    %e = xbg.ref_expr @sym  : %type
    ```
  }];
  let assemblyFormat = [{
    custom<Concept>(type($conceptClass)) $sym_name
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// XBG select expression
//===----------------------------------------------------------------------===//

def XBG_SelectExpr : XBG_Construct<"SelectExprCep", "select_expr", [ExprCep]> {
  let summary = "XBG select expression";
  let description = [{
    Syntax:
    ```mlir
    %e = xbg.select_expr %expr %true_value, %false_value
    ```
  }];
  let args = (ins ExprCep:$expr, ExprCep:$true_value, ExprCep:$false_value);
  let assemblyFormat = [{
    custom<Concept>(type($conceptClass))
    $expr `[` type($expr) `]`
    `true` $true_value `[` type($true_value) `]`
    `false` $false_value `[` type($false_value) `]` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// XBG store expression
//===----------------------------------------------------------------------===//

def XBG_StoreExpr :
    XBG_Construct<"StoreExprCep", "store_expr", [ExprCep]> {
  let summary = "XBG store expression";
  let description = [{
    Syntax:
    ```mlir
    xbg.store_expr %address = %value_to_store
    ```
  }];
  let args = (ins TypeAttrOf<XLG_ConceptType>:$conceptClass,
                  ExprCep:$address,
                  ExprCep:$value);
  let assemblyFormat = [{
    custom<Concept>($conceptClass) $address `[` type($address) `]` `=`
    $value `[` type($value) `]`
    attr-dict
  }];
  let results = (outs);
}

//===----------------------------------------------------------------------===//
// XBG string expressions
//===----------------------------------------------------------------------===//

def XBG_StringExpr : XBG_PureConstruct<"StringExpr", "string_expr", [XBG_ConstExpr]> {
}

//===----------------------------------------------------------------------===//
// XBG TemplateExpr
//===----------------------------------------------------------------------===//

def XBG_TemplateExpr : XBG_Construct<"TemplateExprCep", "template_expr", [
      SymbolCep, ExprCep
    ]> {
  let summary = "XBG template expression";
  let args = (ins ExprCep:$parameter, Optional<ExprCep>:$init);
  let description = [{
    Syntax:
    ```mlir
    %ty = xbg.template_expr @name %arg0 = %i32
    ```
  }];
  let assemblyFormat = [{
     custom<Concept>(type($conceptClass)) $sym_id (`[` $usr^ `]`)?
     $parameter `:` type($parameter) (` ` `=` $init^ `:` type($init))?
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// XBG ToXLGExpr
//===----------------------------------------------------------------------===//

def XBG_ToXLGExpr : XBG_Construct<"ToXLGExprCep", "to_xlg_expr", [ExprCep]> {
  let summary = "Transforms an MLIR value to XLG.";
  let description = [{
    Syntax:
    ```mlir
    %1 = xbg.to_xlg_expr %0 : xlg.class_type<#xbg.expr>
    ```
  }];
  let args = (ins AnyType:$expr);
  let assemblyFormat = [{
     custom<Concept>(type($conceptClass))
    `(` $expr `:` type($expr) `)` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// XBG unary expression
//===----------------------------------------------------------------------===//

def XBG_UnaryExpr : XBG_Construct<"UnaryExprCep", "unary_expr", [ExprCep]> {
  let summary = "XBG unary expression";
  let description = [{
    Syntax:
    ```mlir
    %e = xbg.unary_expr op %expr
    ```
  }];
  let assemblyFormat = [{
    custom<Concept>(type($conceptClass))
   $op $expr `:` type($expr) attr-dict
  }];
  let args = (ins UnaryOperator:$op, ExprCep:$expr);
}

//===----------------------------------------------------------------------===//
// XBG value reference expression
//===----------------------------------------------------------------------===//

def XBG_ValueRefExpr : XBG_Construct<"ValueRefExprCep", "value_ref_expr", [ValueRefExprCep]> {
  let summary = "XBG value reference expression";
  let description = [{
    Syntax:
    ```mlir
    %e = xbg.ref_expr %value
    ```
  }];
  let assemblyFormat = [{
    custom<Concept>(type($conceptClass)) $value `:` type($value)
    attr-dict
  }];
}

#endif // XBLANG_XBG_EXPR_OPS
