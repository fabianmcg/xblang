#ifndef XBLANG_SYNTAX_DIALECT
#define XBLANG_SYNTAX_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// Syntax IR dialect definition.
//===----------------------------------------------------------------------===//

def Syntax_Dialect : Dialect {
  let name = "syntax";
  let summary = "Syntax generator MLIR dialect.";
  let cppNamespace = "::xblang::syntaxgen";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base syntax classes definition.
//===----------------------------------------------------------------------===//

class Syntax_Op<string mnemonic, list<Trait> traits = []>
    : Op<Syntax_Dialect, mnemonic, traits>;

class Syntax_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Syntax_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

class Syntax_Attr<string name, string attrMnemonic, list<Trait> traits = []>:
  AttrDef<Syntax_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// Syntax types definition.
//===----------------------------------------------------------------------===//

def Syntax_ExprType : Syntax_Type<"Expr", "expr"> {
  let summary = "Type for syntax expressions";
  let description = [{
    Syntax:

    ```mlir
    expr ::= `expr`
    ```
  }];
  let assemblyFormat = "";
}

//===----------------------------------------------------------------------===//
// Syntax attr definition.
//===----------------------------------------------------------------------===//

def Syntax_LiteralAttr : Syntax_Attr<"Literal", "literal"> {
  let summary = "Attribute for lexing literals";
  let parameters = (ins AttrParameter<"uint32_t", "Unicode point">:$literal);
  let assemblyFormat = [{ `<` custom<Literal>($literal) `>` }];
  let genVerifyDecl = 0;
}

def Syntax_CharClass : AttrParameter<"CharClass", "lexing character class"> {
  let printer = [{
  	$_printer << '"' << $_self.toString() << '"'
  }];
}

def Syntax_CharClassAttr : Syntax_Attr<"CharClass", "char_class"> {
  let summary = "Attribute for char classes";
  let parameters = (ins Syntax_CharClass:$charClass);
  let assemblyFormat = [{ `<` $charClass `>` }];
  let genVerifyDecl = 0;
}

def Syntax_LexTerminalKindEnum : I32EnumAttr<"LexTerminalKind", "Lex terminal kind", [
    I32EnumAttrCase<"Token", 0, "tok">,
    I32EnumAttrCase<"Class", 1, "class">,
    I32EnumAttrCase<"Unspecified", 2, "unk">,
    I32EnumAttrCase<"Dynamic", 3, "dyn">,
    I32EnumAttrCase<"Any", 4, "any">,
  ]> {
  let cppNamespace = "::xblang::syntaxgen";
}

def Syntax_LexTerminalAttr : Syntax_Attr<"LexTerminal", "lex_terminal"> {
  let summary = "Attribute for lexing terminals";
  let parameters = (ins "::mlir::FlatSymbolRefAttr":$identifier,
                        DefaultValuedParameter<"LexTerminalKind",
                                               "LexTerminalKind::Token">:$kind,
                        StringRefParameter<"", [{""}]>:$alias);
  let assemblyFormat = [{ `<` $identifier (`,` $kind^)? (`,` $alias^)? `>` }];
  let genVerifyDecl = 0;
}

def Syntax_CodeActionAttr : Syntax_Attr<"CodeAction", "code"> {
  let summary = "Attribute for code actions";
  let parameters = (ins StringRefParameter<"", [{""}]>:$pre_action,
                        StringRefParameter<"", [{""}]>:$post_action);
  let assemblyFormat = [{
    `<` struct($pre_action, $post_action)`>`
  }];
  let genVerifyDecl = 0;
}

//===----------------------------------------------------------------------===//
// Syntax module helper class.
//===----------------------------------------------------------------------===//

class SyntaxModule<string mnemonic, list<Trait> traits = []> : Syntax_Op<mnemonic, [
    IsolatedFromAbove, NoRegionArguments, OpAsmOpInterface,
    Symbol, SingleBlock
  ] # GraphRegionNoTerminator.traits # traits> {
  let arguments = (ins OptionalAttr<SymbolNameAttr>:$sym_name,
                       OptionalAttr<StrAttr>:$sym_visibility);
  let regions = (region SizedRegion<1>:$bodyRegion);
  let assemblyFormat = "($sym_name^)? attr-dict-with-keyword $bodyRegion";
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Allow the dialect prefix to be omitted.
    static llvm::StringRef getDefaultDialect() { return "syntax"; }
  }];
}

#endif // XBLANG_SYNTAX_DIALECT
