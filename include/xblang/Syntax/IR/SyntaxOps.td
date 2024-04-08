#ifndef XBLANG_SYNTAX_OPS
#define XBLANG_SYNTAX_OPS

include "xblang/Syntax/IR/SyntaxDialect.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Syntax Rule Op
//===----------------------------------------------------------------------===//

def Syntax_RuleOp : Syntax_Op<"rule", [
     FunctionOpInterface, IsolatedFromAbove, SingleBlock, OpAsmOpInterface
  ]> {
  let summary = "Creates a syntax rule.";
  let description = [{
    Syntax:
    ```mlir
    syntax.rule @myRule {
      ...
      return %expr
    }
    ```
  }];
  let regions = (region SizedRegion<1>:$bodyRegion);
  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<StrAttr>:$sym_visibility,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  let builders = [OpBuilder<(ins "mlir::StringRef":$name)>];
  let assemblyFormat = [{
    $sym_name custom<RuleType>($function_type)
    attr-dict-with-keyword $bodyRegion
  }];
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    mlir::ArrayRef<mlir::Type> getArgumentTypes() {
      return getFunctionType().getInputs();
    }

    /// Returns the result types of this function.
    mlir::ArrayRef<mlir::Type> getResultTypes() {
      return getFunctionType().getResults();
    }

    /// Returns the region on the function operation that is callable.
    mlir::Region *getCallableRegion() { return &getBodyRegion(); }

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Allow the dialect prefix to be omitted.
    static llvm::StringRef getDefaultDialect() { return "syntax"; }
  }];
}

//===----------------------------------------------------------------------===//
// Syntax Macro Op
//===----------------------------------------------------------------------===//

def Syntax_MacroOp : Syntax_Op<"macro", [
    FunctionOpInterface, IsolatedFromAbove, SingleBlock, OpAsmOpInterface,
  ]> {
  let summary = "Creates a syntax macro.";
  let description = [{
    Syntax:
    ```mlir
    syntax.macro @myMacro {
      ...
      return %expr
    }
    ```
  }];
  let regions = (region SizedRegion<1>:$bodyRegion);
  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<StrAttr>:$sym_visibility,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  let builders = [OpBuilder<(ins "mlir::StringRef":$name,
                                 CArg<"size_t", "0">:$numArgs)>];
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    mlir::ArrayRef<mlir::Type> getArgumentTypes() {
      return getFunctionType().getInputs();
    }

    /// Returns the result types of this function.
    mlir::ArrayRef<mlir::Type> getResultTypes() {
      return getFunctionType().getResults();
    }

    /// Returns the region on the function operation that is callable.
    mlir::Region *getCallableRegion() { return &getBodyRegion(); }

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Allow the dialect prefix to be omitted.
    static llvm::StringRef getDefaultDialect() { return "syntax"; }
  }];
}

//===----------------------------------------------------------------------===//
// Syntax Return Op
//===----------------------------------------------------------------------===//

def Syntax_ReturnOp : Syntax_Op<"return", [
	Pure, ParentOneOf<["MacroOp", "RuleOp"]>, ReturnLike, Terminator
  ]> {
  let summary = "Creates a definition return.";
  let description = [{
    Syntax:
    ```mlir
    syntax.return %expr
    ```
  }];
  let arguments = (ins Syntax_ExprType:$expr);
  let assemblyFormat = "$expr attr-dict-with-keyword";
}

//===----------------------------------------------------------------------===//
// Syntax Empty String Op
//===----------------------------------------------------------------------===//

def Syntax_EmptyStringOp : Syntax_Op<"eps", [
    ConstantLike, Pure,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]> {
  let summary = "Creates an empty string constant.";
  let description = [{
    The `syntax.eps` operation produces an expr-typed SSA value for the empty
    string constant.

    Example:

    ```mlir
    %eps = syntax.eps
    ```
  }];
  let results = (outs Syntax_ExprType:$result);
  let assemblyFormat = "attr-dict";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Syntax Zero Or More Op
//===----------------------------------------------------------------------===//

def Syntax_ZeroOrMoreOp : Syntax_Op<"zero_or_more", [
    Pure, Idempotent, SameOperandsAndResultType,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]> {
  let summary = "Creates a zero or more expression.";
  let description = [{
    The `syntax.zero_or_more` operation produces a zero or more expression.

    Example:

    ```mlir
    %zom = syntax.zero_or_more %expr
    ```
  }];
  let arguments = (ins Syntax_ExprType:$expr,
                       OptionalAttr<ArrayAttr>:$first_set,
                       UnitAttr:$nullable);
  let results = (outs Syntax_ExprType:$result);
  let assemblyFormat = [{
    $expr (`first_set` `=` $first_set^)? attr-dict
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Syntax And Op
//===----------------------------------------------------------------------===//

def Syntax_AndOp : Syntax_Op<"and", [
    Pure,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]> {
  let summary = "Creates an and expression.";
  let description = [{
    The `syntax.and` operation produces the logical and expression between two
    expressions.

    Example:

    ```mlir
    %and = syntax.and %expr1, %expr2
    ```
  }];
  let arguments = (ins Syntax_ExprType:$LHS, Syntax_ExprType:$RHS);
  let results = (outs Syntax_ExprType:$result);
  let assemblyFormat = "$LHS `,` $RHS  attr-dict";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Syntax Or Op
//===----------------------------------------------------------------------===//

def Syntax_OrOp : Syntax_Op<"or", [
    Pure,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]> {
  let summary = "Creates an or expression.";
  let description = [{
    The `syntax.or` operation produces the logical or expression between two
    expressions.

    Example:

    ```mlir
    %or = syntax.or %expr1, %expr2
    ```
  }];
  let arguments = (ins Syntax_ExprType:$LHS, Syntax_ExprType:$RHS);
  let results = (outs Syntax_ExprType:$result);
  let assemblyFormat = "$LHS `,` $RHS  attr-dict";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Syntax Terminal Op
//===----------------------------------------------------------------------===//

def Syntax_TerminalAttr : AnyAttrOf<[
    Syntax_LiteralAttr, Syntax_CharClassAttr, Syntax_LexTerminalAttr
  ]>;

def Syntax_TerminalOp : Syntax_Op<"terminal", [
    Pure, ConstantLike,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]> {
  let summary = "Creates a terminal expression.";
  let description = [{
    The `syntax.terminal` operation produces a reference to a terminal symbol.

    Example:

    ```mlir
    %terminal = syntax.terminal #syntax.literal<"literal">
    ```
  }];
  let arguments = (ins Syntax_TerminalAttr:$terminal);
  let results = (outs Syntax_ExprType:$result);
  let assemblyFormat = "$terminal attr-dict";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Syntax Call Op
//===----------------------------------------------------------------------===//

def Syntax_CallOp : Syntax_Op<"call", [
    Pure,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    DeclareOpInterfaceMethods<CallOpInterface>
  ]> {
  let summary = "Creates a call expression.";
  let description = [{
    The `syntax.ref` operation produces a call to a macro.

    Example:

    ```mlir
    %ref1 = syntax.call @mySyntaxRule
    ```
  }];
  let arguments = (ins FlatSymbolRefAttr:$callee,
                       Variadic<Syntax_ExprType>:$operands);
  let builders = [OpBuilder<(ins "mlir::StringRef":$name)>];
  let results = (outs Syntax_ExprType:$result);
  let assemblyFormat = "$callee (`(` $operands^ `)`)? attr-dict";
}

//===----------------------------------------------------------------------===//
// Syntax Metadata Op
//===----------------------------------------------------------------------===//

def Syntax_MetadataOp : Syntax_Op<"md_node", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]> {
  let summary = "Creates an metadata expression.";
  let description = [{
    The `syntax.md_node` operation produces a non-removable expression to hold
    extra-information.

    Example:

    ```mlir
    %md1 = syntax.md_node %node
    ```
  }];
  let arguments = (ins Syntax_ExprType:$expr,
                       OptionalAttr<FlatSymbolRefAttr>:$name,
                       OptionalAttr<Syntax_CodeActionAttr>:$codeAction);
  let results = (outs Syntax_ExprType:$result);
  let assemblyFormat = [{
    $expr ($name^)? (`action` `=` $codeAction^)? attr-dict
  }];
  let hasFolder = 1;
  // MD nodes cannot be sandwiched by Or-Or or And-And
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Syntax lexing ops
//===----------------------------------------------------------------------===//

include "xblang/Syntax/IR/SyntaxLexOps.td"

//===----------------------------------------------------------------------===//
// Syntax parsing ops
//===----------------------------------------------------------------------===//

include "xblang/Syntax/IR/SyntaxParseOps.td"

#endif // XBLANG_SYNTAX_OPS
