#ifndef XBLANG_TYPE_SYNTAX
#define XBLANG_TYPE_SYNTAX

include "xblang/Syntax/SyntaxBase.td"

//===----------------------------------------------------------------------===//
// XB parse type
//===----------------------------------------------------------------------===//

def XB_TypeStx : Production<"Type", "::mlir::Value"> {
  let rule = [{
    {{ ::mlir::Value type; }}
    /*parse the base unqualified type*/
    UnqualifiedType {{ type = _0.get(); }}
    /*parse type decorators*/
    (
      /*parse pointer decorator*/
      "*":$ptrTk {{
        type = create<PointerType>(getLoc(ptrTk),
                                   getConceptClass<PointerTypeCep>(),
                                   nullptr,
                                   type);
      }} | {{ ValueList dims; }}
      /*parse array decorator*/
      "[":$lBracket @Interleave(Expr:$expr {{ dims.push_back(expr.get()); }}, ",") "]"
        {{
          type = create<ArrayType>(getLoc(lBracket),
                                   getConceptClass<ArrayTypeCep>(),
                                   nullptr,
                                   type,
                                   dims);
        }}
    )*
    ( /*parse a reference decorator*/
      "&":$refTok {{
        type = create<ReferenceType>(getLoc(refTok),
                                     getConceptClass<ReferenceTypeCep>(),
                                     nullptr,
                                     type);
      }}
    )?
    {{
      $res = type;
    }}
  }];
}

//===----------------------------------------------------------------------===//
// XB parse unqualified types
//===----------------------------------------------------------------------===//

def XB_UnqualifiedTypeStx : Production<"UnqualifiedType", "::mlir::Value"> {
  let rule = [{
    BuiltinType {{
      $res = _0.get();
    }} |
    #dyn_kw("xlg::Type") {{
      $res = _0.get();
    }} |
    QualifiedIdentifierType {{
      $res = _0.get();
    }}
  }];
}

//===----------------------------------------------------------------------===//
// XB parse builtin types
//===----------------------------------------------------------------------===//

def XB_BuiltinTypeStx : Production<"BuiltinType", "::mlir::Value"> {
  let rule = [{
    Integer {{ 
      $res = _0.get();
    }} |
    FloatType {{ 
      $res = _0.get();
    }} |
    "bool" {{
      $res = create<BuiltinType>(getLoc(_0),
                                 getConceptClass<BuiltinTypeCep>(),
                                 getI1Type());
    }}
  }];
}

//===----------------------------------------------------------------------===//
// XB parse integer
//===----------------------------------------------------------------------===//

def XB_IntegerStx : Production<"Integer", "::mlir::Value"> {
  let rule = [{
    "i8" {{
      $res = create<BuiltinType>(getLoc(_0),
                                 getConceptClass<BuiltinTypeCep>(),
                                 getIntegerType(8, true));
    }} |
    "i16" {{
      $res = create<BuiltinType>(getLoc(_0),
                                 getConceptClass<BuiltinTypeCep>(),
                                 getIntegerType(16, true));
    }} |
    "i32" {{
      $res = create<BuiltinType>(getLoc(_0),
                                 getConceptClass<BuiltinTypeCep>(),
                                 getIntegerType(32, true));
    }} |
    "i64" {{
      $res = create<BuiltinType>(getLoc(_0),
                                 getConceptClass<BuiltinTypeCep>(),
                                 getIntegerType(64, true));
    }} |
    "u8" {{
      $res = create<BuiltinType>(getLoc(_0),
                                 getConceptClass<BuiltinTypeCep>(),
                                 getIntegerType(8, false));
    }} |
    "u16" {{
      $res = create<BuiltinType>(getLoc(_0),
                                 getConceptClass<BuiltinTypeCep>(),
                                 getIntegerType(16, false));
    }} |
    "u32" {{
      $res = create<BuiltinType>(getLoc(_0),
                                 getConceptClass<BuiltinTypeCep>(),
                                 getIntegerType(32, false));
    }} |
    "u64" {{
      $res = create<BuiltinType>(getLoc(_0),
                                 getConceptClass<BuiltinTypeCep>(),
                                 getIntegerType(64, false));
    }} |
    "uint" {{
      $res = create<BuiltinType>(getLoc(_0),
                                 getConceptClass<BuiltinTypeCep>(),
                                 getIndexType());
    }}
  }];
}

//===----------------------------------------------------------------------===//
// XB parse float
//===----------------------------------------------------------------------===//

def XB_FloatStx : Production<"FloatType", "::mlir::Value"> {
  let rule = [{
    "f16" {{
      $res = create<BuiltinType>(getLoc(_0),
                                 getConceptClass<BuiltinTypeCep>(),
                                 getF16Type());
    }} |
    "f32" {{
      $res = create<BuiltinType>(getLoc(_0),
                                 getConceptClass<BuiltinTypeCep>(),
                                 getF32Type());
    }} |
    "f64" {{
      $res = create<BuiltinType>(getLoc(_0),
                                 getConceptClass<BuiltinTypeCep>(),
                                 getF64Type());
    }}
  }];
}

//===----------------------------------------------------------------------===//
// XB parse a qualified identifier type
//===----------------------------------------------------------------------===//

def XB_QualifiedIdentifierTypeStx :
    Production<"QualifiedIdentifierType", "::mlir::Value"> {
  let implement = 0;
  let rule = [{
    ("::")? Identifier
  }];
}

#endif // XBLANG_TYPE_SYNTAX
