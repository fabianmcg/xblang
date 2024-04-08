#ifndef XBLANG_SYNTAX_LEX_OPS
#define XBLANG_SYNTAX_LEX_OPS

include "xblang/Syntax/IR/SyntaxDialect.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Syntax Lexer Op
//===----------------------------------------------------------------------===//

def Syntax_LexerOp : SyntaxModule<"lexer", [SymbolTable]> {
  let summary = "Creates a lexer module.";
  let description = [{
    Syntax:
    ```mlir
    syntax.lexer @myLexer {
      ...
    }
    ```
  }];
}

//===----------------------------------------------------------------------===//
// Syntax DFA Op
//===----------------------------------------------------------------------===//

def Syntax_DFAOp : SyntaxModule<"dfa", [SymbolTable]> {
  let summary = "Creates a dfa module.";
  let description = [{
    Syntax:
    ```mlir
    syntax.dfa @myDFA {
      ...
    }
    ```
  }];
}

//===----------------------------------------------------------------------===//
// Syntax Lexing Rule Op
//===----------------------------------------------------------------------===//

def Syntax_LexRuleOp : Syntax_Op<"lex_rule", [
    Symbol
  ]> {
  let summary = "Creates a lexing rule.";
  let description = [{
    The `syntax.lex_rule` operation produces a lexing rule expression.

    Example:

    ```mlir
    syntax.lex_rule @myRule %expr
    ```
   }];
  let arguments = (ins Syntax_ExprType:$expr, SymbolNameAttr:$sym_name);
  let assemblyFormat = "$sym_name $expr attr-dict-with-keyword";
}

//===----------------------------------------------------------------------===//
// Syntax Lex state Op
//===----------------------------------------------------------------------===//

def Syntax_LexStateOp : Syntax_Op<"lex_state", [
    NoRegionArguments, OpAsmOpInterface, Symbol, SingleBlock
  ] # GraphRegionNoTerminator.traits> {
  let summary = "Creates a lexer state.";
  let description = [{
    Syntax:
    ```mlir
    syntax.lex_state @myState {
      ...
    }
    ```
  }];
  let regions = (region AnyRegion:$bodyRegion);
  let arguments = (ins SymbolNameAttr:$sym_name,
                       UnitAttr:$final_state,
                       OptionalAttr<SymbolNameAttr>:$id);
  let assemblyFormat = [{
    (`final` $final_state^)? $sym_name (`id` `=` $id^)? attr-dict-with-keyword
    ($bodyRegion^)?
  }];
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Allow the dialect prefix to be omitted.
    static llvm::StringRef getDefaultDialect() { return "syntax"; }
  }];
}

//===----------------------------------------------------------------------===//
// Syntax Lex Transition Op
//===----------------------------------------------------------------------===//

def Syntax_LexTransitionOp : Syntax_Op<"lex_transition", []> {
  let summary = "Creates a lexing state transition.";
  let description = [{
    The `syntax.lex_transition` operation produces a state transition.

    Example:

    ```mlir
    syntax.lex_transition %terminal -> @nextState
    ```
  }];
  let arguments = (ins Syntax_ExprType:$terminal,
                       FlatSymbolRefAttr:$nextState);
  let assemblyFormat = "$terminal `->` $nextState attr-dict";
}

#endif // XBLANG_SYNTAX_LEX_OPS
