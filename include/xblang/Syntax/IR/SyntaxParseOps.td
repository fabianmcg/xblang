#ifndef XBLANG_SYNTAX_PARSE_OPS
#define XBLANG_SYNTAX_PARSE_OPS

include "xblang/Syntax/IR/SyntaxDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Syntax Parser Op
//===----------------------------------------------------------------------===//

def Syntax_ParserOp : SyntaxModule<"parser", [SymbolTable]> {
  let summary = "Creates a parser module.";
  let description = [{
    Syntax:
    ```mlir
    syntax.parser @myParser {
      ...
    }
    ```
  }];
  let arguments = (ins FlatSymbolRefAttr:$startSymbol,
                       OptionalAttr<SymbolNameAttr>:$sym_name,
                       OptionalAttr<StrAttr>:$sym_visibility);
  let assemblyFormat = [{
    ($sym_name^)? `start` `=` $startSymbol attr-dict-with-keyword $bodyRegion
  }];
}

//===----------------------------------------------------------------------===//
// Syntax NonTerminal Op
//===----------------------------------------------------------------------===//

def Syntax_NonTerminalOp : Syntax_Op<"non_terminal", [
    Pure, ConstantLike,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]> {
  let summary = "Creates a non-terminal expression.";
  let description = [{
    The `syntax.non_terminal` operation produces a reference to a non-terminal.

    Example:

    ```mlir
    %non_terminal = syntax.non_terminal @nonTerminal
    ```
  }];
  let arguments = (ins FlatSymbolRefAttr:$nonTerminal,
                       OptionalAttr<StrAttr>:$dynamic);
  let results = (outs Syntax_ExprType:$result);
  let assemblyFormat = "$nonTerminal (`dynamic` $dynamic^)? attr-dict";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Syntax Switch Op
//===----------------------------------------------------------------------===//

def Syntax_SwitchOp : Syntax_Op<"switch", [
    Pure,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]> {
  let summary = "Creates a switch expression.";
  let description = [{
    The `syntax.switch` operation produces a branch to a series expressions.
    Switches are always predictable, for non predictable branches use `any`.

    Example:

    ```mlir
    %and = syntax.switch %expr1, %expr2, %expr3
    ```
  }];
  let arguments = (ins Variadic<Syntax_ExprType>:$alternatives,
                       ArrayAttr:$first_sets,
                       UnitAttr:$nullable);
  let results = (outs Syntax_ExprType:$result);
  let assemblyFormat = [{
    $alternatives `first_sets` `=` $first_sets attr-dict
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Syntax Sequence Op
//===----------------------------------------------------------------------===//

def Syntax_SequenceOp : Syntax_Op<"seq", [
    Pure,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]> {
  let summary = "Creates a sequence expression.";
  let description = [{
    The `syntax.seq` operation produces a branch to a series expressions.

    Example:

    ```mlir
    %and = syntax.seq %expr1, %expr2, %expr3
    ```
  }];
  let arguments = (ins Variadic<Syntax_ExprType>:$alternatives);
  let results = (outs Syntax_ExprType:$result);
  let assemblyFormat = "$alternatives  attr-dict";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Syntax Any Op
//===----------------------------------------------------------------------===//

def Syntax_AnyOp : Syntax_Op<"any", [
    Pure,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]> {
  let summary = "Creates an any expression.";
  let description = [{
    The `syntax.any` operation produces a branch to a series expressions.

    Example:

    ```mlir
    %any = syntax.any %expr1, %expr2, %expr3
    ```
  }];
  let arguments = (ins Variadic<Syntax_ExprType>:$alternatives,
                       ArrayAttr:$first_sets,
                       ArrayAttr:$conflicts,
                       UnitAttr:$nullable);
  let results = (outs Syntax_ExprType:$result);
  let assemblyFormat = [{
    $alternatives `first_sets` `=` $first_sets `conflicts` `=` $conflicts
    attr-dict
  }];
  let hasFolder = 1;
}


#endif // XBLANG_SYNTAX_PARSE_OPS
