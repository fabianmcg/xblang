#ifndef XBLANG_XBG_DECL_OPS
#define XBLANG_XBG_DECL_OPS

include "xblang/Lang/XBLang/XLG/XBGDialect.td"
include "xblang/XLG/Concepts.td"
include "mlir/IR/RegionKindInterface.td"

//===----------------------------------------------------------------------===//
// XBG FuncDecl
//===----------------------------------------------------------------------===//

def XBG_FuncDecl : XBG_Construct<"FuncDeclCep", "func_decl", [FuncDeclCep], [
      DeclareOpInterfaceMethods<SymbolInterface, ["getSymbolProps"]>,
      DeclareOpInterfaceMethods<SymbolTableInterface, ["getSymbolTableKind"]>,
      SingleBlock
    ]> {
  let summary = "XBG function declaration";
  let description = [{
    Syntax:
    ```mlir
    %fd = xbg.func_def @name %type {
      ...
    }
    ```
  }];
  let regions = (region MaxSizedRegion<1>:$bodyRegion);
  let assemblyFormat = [{
    custom<Concept>(type($conceptClass)) $sym_id (`[` $usr^ `]`)?
    (`type` $type^)?
    attr-dict-with-keyword ($bodyRegion^)?
  }];
}

//===----------------------------------------------------------------------===//
// XBG FuncDef
//===----------------------------------------------------------------------===//

def XBG_FuncDef : XBG_Construct<"FuncDefCep", "func_def", [FuncDefCep],
    GraphRegionNoTerminator.traits> {
  let summary = "XBG function definition";
  let description = [{
    Syntax:
    ```mlir
    %vd = xbg.func_def %retTy (%args...)
    ```
  }];
  let regions = (region MaxSizedRegion<1>:$bodyRegion);
  let assemblyFormat = [{
    custom<Concept>(type($conceptClass))
    `(` $arguments `)` ( `->` $returnType^)? `:`
    functional-type($arguments, $returnType)
    attr-dict-with-keyword ($bodyRegion^)?
  }];
}

//===----------------------------------------------------------------------===//
// XBG ModuleDecl
//===----------------------------------------------------------------------===//

def XBG_ModuleDecl : XBG_Construct<"ModuleDeclCep", "module_decl",
    [ModuleDeclCep], GraphRegionNoTerminator.traits> {
  let summary = "XBG module declaration";
  let description = [{
    Syntax:
    ```mlir
    xbg.module_decl @name {
      ...
    }
    ```
  }];
  let args = (ins TypeAttrOf<XLG_ConceptType>:$conceptClass);
  let regions = (region MaxSizedRegion<1>:$bodyRegion);
  let assemblyFormat = [{
    custom<Concept>($conceptClass) $sym_id
    attr-dict-with-keyword $bodyRegion
  }];
  let results = (outs);
}

//===----------------------------------------------------------------------===//
// XBG ObjectDecl
//===----------------------------------------------------------------------===//

def XBG_ObjectDecl : XBG_Construct<"ObjectDeclCep", "object_decl",
    [ObjectDeclCep], GraphRegionNoTerminator.traits> {
  let summary = "XBG object declaration";
  let description = [{
    Syntax:
    ```mlir
    %vd = xbg.object_decl @name %type {
      ...
    }
    ```
  }];
  let regions = (region MaxSizedRegion<1>:$declBody);
  let assemblyFormat = [{
    custom<Concept>(type($conceptClass)) $sym_id
    (`[` $type^ `]`)?
    attr-dict-with-keyword ($declBody^)?
  }];
}

//===----------------------------------------------------------------------===//
// XBG VarDecl
//===----------------------------------------------------------------------===//

def XBG_VarDecl : XBG_Construct<"VarDeclCep", "var_decl", [VarDeclCep]> {
  let summary = "XBG variable declaration";
  let description = [{
    Syntax:
    ```mlir
    %vd = xbg.var_decl @name %var_type : type(var_type) = %init `:` type(init)
    ```
  }];
  let assemblyFormat = [{
    custom<Concept>(type($conceptClass)) $sym_id
    ($valueType^ `:` type($valueType))? (`[` $type^ `]`)?
    (`=` $expr^ `:` type($expr))? attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// XBG ParamDecl
//===----------------------------------------------------------------------===//

def XBG_ParamDecl : XBG_PureConstruct<"ParamDecl", "param_decl", [VarDeclCep]> {
}

//===----------------------------------------------------------------------===//
// XBG MemberDecl
//===----------------------------------------------------------------------===//

def XBG_MemberDecl : XBG_PureConstruct<"MemberDecl", "member_decl", [VarDeclCep]> {
}

#endif // XBLANG_XBG_DECL_OPS
