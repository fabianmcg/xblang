#ifndef XBLANG_XBG_TYPE_OPS
#define XBLANG_XBG_TYPE_OPS

include "xblang/Lang/XBLang/XLG/XBGDialect.td"
include "xblang/XLG/Concepts.td"

//===----------------------------------------------------------------------===//
// XBG BuiltinType
//===----------------------------------------------------------------------===//

def XBG_BuiltinType :
    XBG_Construct<"BuiltinTypeCep", "builtin_type", [BuiltinTypeCep]> {
  let summary = "XBG builtin type";
  let description = [{
    Syntax:
    ```mlir
    %ty = xbg.builtin_type %type
    ```
  }];
  let assemblyFormat = [{
     custom<Concept>(type($conceptClass)) $type attr-dict
  }];
  let builders = [
    OpBuilder<(ins "::mlir::Type":$conceptClass, "::mlir::Type":$type)>
  ];
}

//===----------------------------------------------------------------------===//
// XBG ArrayType
//===----------------------------------------------------------------------===//

def XBG_ArrayType :
    XBG_Construct<"ArrayTypeCep", "array_type", [DerivedTypeCep]> {
  let summary = "XBG array type";
  let args = (ins TypeCep:$base, Variadic<ExprCep>:$shape);
  let description = [{
    Syntax:
    ```mlir
    %ty = xbg.array_type %base [...]
    ```
  }];
  let assemblyFormat = [{
     custom<Concept>(type($conceptClass)) $base `:` type($base) ` `
     `[` $shape `]` `:` `(` type($shape) `)`
     attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// XBG PointerType
//===----------------------------------------------------------------------===//

def XBG_PointerType :
    XBG_Construct<"PointerTypeCep", "ptr_type", [DerivedTypeCep]> {
  let summary = "XBG pointer type";
  let args = (ins TypeCep:$base);
  let description = [{
    Syntax:
    ```mlir
    %ty = xbg.ptr_type %type
    ```
  }];
  let assemblyFormat = [{
     custom<Concept>(type($conceptClass)) $base `:` type($base) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// XBG ReferenceType
//===----------------------------------------------------------------------===//

def XBG_ReferenceType :
    XBG_Construct<"ReferenceTypeCep", "ref_type", [DerivedTypeCep]> {
  let summary = "XBG reference type";
  let args = (ins TypeCep:$base);
  let description = [{
    Syntax:
    ```mlir
    %ty = xbg.ref_type %type
    ```
  }];
  let assemblyFormat = [{
     custom<Concept>(type($conceptClass)) $base `:` type($base) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// XBG RefExprType
//===----------------------------------------------------------------------===//

def XBG_RefExprType :
    XBG_Construct<"RefExprTypeCep", "ref_expr_type", [DerivedTypeCep]> {
  let summary = "XBG reference expression type";
  let args = (ins SymbolRefAttr:$sym_name);
  let description = [{
    Syntax:
    ```mlir
    %ty = xbg.ref_expr_type @name
    ```
  }];
  let assemblyFormat = [{
     custom<Concept>(type($conceptClass)) $sym_name attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// XBG RemoveReferenceType
//===----------------------------------------------------------------------===//

def XBG_RemoveReferenceType :
    XBG_Construct<"RemoveReferenceTypeCep", "rm_ref_type", [TypeCep]> {
  let summary = "XBG removes a reference from a type";
  let args = (ins TypeCep:$base);
  let description = [{
    Syntax:
    ```mlir
    %ty = xbg.rm_ref_type %base
    ```
  }];
  let assemblyFormat = [{
     custom<Concept>(type($conceptClass)) $base `:` type($base) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// XBG TemplateType
//===----------------------------------------------------------------------===//

def XBG_TemplateType : XBG_Construct<"TemplateTypeCep", "template_type", [
      SymbolCep, TypeCep
    ]> {
  let summary = "XBG template type";
  let args = (ins TypeCep:$parameter, Optional<TypeCep>:$init);
  let description = [{
    Syntax:
    ```mlir
    %ty = xbg.template_type @name %arg0 = %i32
    ```
  }];
  let assemblyFormat = [{
     custom<Concept>(type($conceptClass)) $sym_id (`[` $usr^ `]`)?
     $parameter `:` type($parameter) (` ` `=` $init^ `:` type($init))?
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// XBG TypeOf
//===----------------------------------------------------------------------===//

def XBG_TypeOf :
    XBG_Construct<"TypeOfCep", "type_of", [TypeCep]> {
  let summary = "XBG type of construct";
  let args = (ins XLG_ConceptType:$expr);
  let description = [{
    Syntax:
    ```mlir
    %ty = xbg.type_of %expr
    ```
  }];
  let assemblyFormat = [{
     custom<Concept>(type($conceptClass)) $expr `:` type($expr) attr-dict
  }];
}

#endif // XBLANG_XBG_TYPE_OPS
