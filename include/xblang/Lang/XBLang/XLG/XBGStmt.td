#ifndef XBLANG_XBG_STMT_OPS
#define XBLANG_XBG_STMT_OPS

include "xblang/Lang/XBLang/XLG/XBGDialect.td"
include "xblang/XLG/Concepts.td"
include "xblang/Dialect/XBLang/IR/Enums.td"
include "mlir/IR/RegionKindInterface.td"

//===----------------------------------------------------------------------===//
// XBG Stmt
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// XBG CompoundStmt
//===----------------------------------------------------------------------===//

def XBG_CompoundStmt : XBG_Construct<"CompoundStmtCep", "compound_stmt",
    [StmtCep], GraphRegionNoTerminator.traits> {
  let summary = "XBG compound statement";
  let description = [{
    Syntax:
    ```mlir
    xbg.compound_stmt {
      // Statement list
    }
    ```
  }];
  let args = (ins TypeAttrOf<XLG_ConceptType>:$conceptClass);
  let regions = (region SizedRegion<1>:$bodyRegion);
  let assemblyFormat = [{
    custom<Concept>($conceptClass) attr-dict-with-keyword $bodyRegion
  }];
  let results = (outs);
}

//===----------------------------------------------------------------------===//
// XBG IfStmt
//===----------------------------------------------------------------------===//

def XBG_IfStmt : XBG_Construct<"IfStmtCep", "if_stmt",
    [ControlFlowStmtCep], GraphRegionNoTerminator.traits> {
  let summary = "XBG If statement";
  let description = [{
    Syntax:
    ```mlir
    xbg.if_stmt (%condition) {
      // Statement list
    }
    ```
  }];
  let args = (ins TypeAttrOf<XLG_ConceptType>:$conceptClass,
                  ExprCep:$condition);
  let regions = (region SizedRegion<1>:$thenRegion,
                        MaxSizedRegion<1>:$elseRegion);
  let assemblyFormat = [{
    custom<Concept>($conceptClass)
    `(` $condition `:` type($condition) `)` attr-dict-with-keyword
    `then` $thenRegion
    (`else` $elseRegion^)?
  }];
  let results = (outs);
}

//===----------------------------------------------------------------------===//
// XBG RangeForStmt
//===----------------------------------------------------------------------===//

def XBG_RangeForStmt : XBG_Construct<"RangeForStmtCep", "range_for_stmt",
    [ControlFlowStmtCep],
    GraphRegionNoTerminator.traits # [AttrSizedOperandSegments]> {
  let summary = "XBG range for statement";
  let description = [{
    Syntax:
    ```mlir
    xbg.range_for_stmt (...) {
      ...
    }
    ```
  }];
  let args = (ins TypeAttrOf<XLG_ConceptType>:$conceptClass,
                  Variadic<RefExprCep>:$iterators,
                  Variadic<XLGConceptClass<"::xblang::xbg::RangeExprCep", "`range expression`">>:$ranges);
  let regions = (region SizedRegion<1>:$bodyRegion);
  let assemblyFormat = [{
    custom<Concept>($conceptClass)
    `(` $iterators`[` type($iterators) `]` `in`$ranges `[`type($ranges)`]` `)`
    attr-dict-with-keyword $bodyRegion
  }];
  let results = (outs);
}

//===----------------------------------------------------------------------===//
// XBG ReturnStmt
//===----------------------------------------------------------------------===//

def XBG_ReturnStmt : XBG_Construct<"ReturnStmtCep", "return",
    [ReturnStmtCep]> {
  let summary = "XBG return statement";
  let description = [{
    Syntax:
    ```mlir
    xbg.return %x : expr
    ```
  }];
  let args = (ins TypeAttrOf<XLG_ConceptType>:$conceptClass,
                  Optional<ExprCep>:$expr);
  let assemblyFormat = [{
    custom<Concept>($conceptClass) ($expr^ `:` type($expr) )? attr-dict
  }];
  let results = (outs);
}

//===----------------------------------------------------------------------===//
// XBG WhileStmt
//===----------------------------------------------------------------------===//

def XBG_WhileStmt : XBG_Construct<"WhileStmtCep", "while_stmt",
    [LoopStmtCep], GraphRegionNoTerminator.traits> {
  let summary = "XBG while statement";
  let description = [{
    Syntax:
    ```mlir
    xbg.while (%condition) {
      // Statement list
    }
    ```
  }];
  let args = (ins TypeAttrOf<XLG_ConceptType>:$conceptClass,
                  ExprCep:$condition);
  let regions = (region SizedRegion<1>:$bodyRegion);
  let assemblyFormat = [{
    custom<Concept>($conceptClass)
    `(` $condition `:` type($condition) `)` attr-dict-with-keyword
    $bodyRegion
  }];
  let results = (outs);
}

//===----------------------------------------------------------------------===//
// XBG YieldStmt
//===----------------------------------------------------------------------===//

def XBG_YieldStmt : XBG_Construct<"YieldStmtCep", "yield",
    [CFTerminatorStmtCep]> {
  let summary = "XBG yield statement";
  let description = [{
    Syntax:
    ```mlir
    xbg.yield yield_kind
    ```
  }];
  let args = (ins TypeAttrOf<XLG_ConceptType>:$conceptClass, YieldKind:$kind);
  let assemblyFormat = [{
    custom<Concept>($conceptClass) $kind attr-dict
  }];
  let results = (outs);
}

#endif // XBLANG_XBG_STMT_OPS
