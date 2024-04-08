#ifndef XBLANG_DECL_SYNTAX
#define XBLANG_DECL_SYNTAX

include "xblang/Syntax/SyntaxBase.td"

//===----------------------------------------------------------------------===//
// XB helper class for variable-like declarations
//===----------------------------------------------------------------------===//

class XB_VarLikeStx<string declTy, string concept> {
  string baseRule = [{
    {{ mlir::Value type; Value expr; }}
    Identifier (":" Type:$ty {{ type = ty.get(); }})?
    ( "=" Expr:$e {{ expr = e.get(); }} )?
    {{
      $res = create<$declTy>(getLoc(_0),
                             getConceptClass<$concept>(),
                             _0.getSpelling(),
                             /*USR=*/nullptr,
                             nullptr,
                             type,
                             expr);
    }}
  }];
  string rule = !subst("$declTy",
                        declTy,
                        !subst("$concept", concept, baseRule));
}

//===----------------------------------------------------------------------===//
// XB parse a top declaration
//===----------------------------------------------------------------------===//

def XB_TopDeclStx : Production<"TopDecl", "::mlir::Operation*"> {
  let rule = [{
    #dyn_kw("xlg::Decl") {{
      $res = _0.get();
    }}  |
    FuncDecl {{
      $res = _0.get();
    }} |
    TemplateDecl {{
      $res = _0.get();
    }} |
    StructDecl {{
      $res = _0.get();
    }}
  }];
}

//===----------------------------------------------------------------------===//
// XB parse function declarations
//===----------------------------------------------------------------------===//

def XB_FuncAttrsStx : Production<"FuncAttrs", "::mlir::DictionaryAttr"> {
  let rule = [{ {{ SmallVector<mlir::NamedAttribute> attrs; }}
    ("["@Interleave(
      "inline" {{
        attrs.push_back(mlir::NamedAttribute(getStringAttr("inline"),
                                             getUnitAttr()));
      }} |
      "extern" {{
        attrs.push_back(mlir::NamedAttribute(getStringAttr("extern"),
                                             getUnitAttr()));
      }} |
      "target" "=" Identifier:$id {{
        attrs.push_back(mlir::NamedAttribute(getStringAttr("target"),
                                             getStringAttr(id.getSpelling())));
      }} |
      Identifier:$id "=" #dyn_kw("xlg::Attr"):$attr {{
        attrs.push_back(mlir::NamedAttribute(getStringAttr(id.getSpelling()),
                                             attr.get()));
      }}, ",")
    "]")? {{
      if (attrs.size() > 0)
        $res = getDictionaryAttr(attrs);
      else
       $res = nullptr;
    }}
  }];
}

def XB_FuncDeclStx : Production<"FuncDecl", "::mlir::Operation*"> {
  let rule = [{
    /*rule header*/
    {{
      InsertionBlock functionBlock;
    }}
    /*parse the header*/
    "fn" FuncAttrs:$attrs
    Identifier:$id {{functionBlock = getInsertionBlock();}}
    FuncDef:$def
    {{
      auto _loc = getLoc(_0);
      create<::xblang::xlg::ReturnOp>(_loc, def.get());
      functionBlock.restorePoint();
      auto fnDecl = create<FuncDecl>(_loc, getConceptClass<FuncDeclCep>(),
                                     id.getSpelling(), nullptr, nullptr);
      if (attrs.get())
        fnDecl->setDiscardableAttrs(attrs.get());
      if (mlir::Block* block = functionBlock.release()) 
        fnDecl.getBodyRegion().push_back(block);
      $res = fnDecl;
    }}
  }];
}

def XB_FuncDefStx : Production<"FuncDef", "::xblang::xbg::FuncDef"> {
  let rule = [{
    /*rule header*/
    {{
      mlir::Value retTy; mlir::SmallVector<mlir::Value> args;
      InsertionBlock functionBlock;
    }}
    /*parse the arguments*/
    "(" @Interleave(ParamDecl:$decl {{ args.push_back(decl.get()); }}, ",") ")"

    /*parse the return type*/
    ("->" Type:$ty {{ retTy = ty.get(); }})?

    /*parse the body*/
    (
      "{" /*parse a function definition*/
        (
          {{functionBlock = getInsertionBlock();}}
          Stmt*
        )
      "}" |
      ";"
    )
    {{
      auto _loc = getLoc(_0);
      functionBlock.restorePoint();
      auto fnDef = create<FuncDef>(_loc, getConceptClass<FuncDefCep>(), nullptr,
                                   retTy, args);
      if (mlir::Block* block = functionBlock.release()) 
        fnDef.getBodyRegion().push_back(block);
      $res = fnDef;
    }}
  }];
}

//===----------------------------------------------------------------------===//
// XB parse member declarations
//===----------------------------------------------------------------------===//

def XB_MemberDeclStx : Production<"MemberDecl", "::mlir::Value"> {
  let rule = XB_VarLikeStx<"VarDecl", "MemberDecl">.rule;
}
//===----------------------------------------------------------------------===//
// XB parse parameter declarations
//===----------------------------------------------------------------------===//

def XB_ParamDeclStx : Production<"ParamDecl", "::mlir::Value"> {
  let rule = XB_VarLikeStx<"VarDecl", "ParamDecl">.rule;
}

//===----------------------------------------------------------------------===//
// XB parse struct declarations
//===----------------------------------------------------------------------===//

def XB_StructDeclStx : Production<"StructDecl", "::mlir::Operation*"> {
  let rule = [{
    /*rule header*/
    {{
      InsertionBlock structBlock;
    }}
    /*parse the header*/
    "struct" Identifier:$id {{ structBlock = getInsertionBlock(); }}

    /*parse the body*/
    (
      "{" /*parse a struct definition*/
        (
          (TopDecl | MemberDecl ";")*
        )
      "}" |
      ";"
    )
    {{
      structBlock.restorePoint();
      auto decl = create<ObjectDecl>(getLoc(_0),
                                     getConceptClass<ObjectDeclCep>(),
                                     id.getSpelling(),
                                     nullptr,
                                     nullptr
                                    );
      if (mlir::Block* block = structBlock.release())
        decl.getDeclBody().push_back(block);
      $res = decl;
    }}
  }];
}


//===----------------------------------------------------------------------===//
// XB parse template declarations
//===----------------------------------------------------------------------===//

def XB_TemplateDeclStx : Production<"TemplateDecl", "::mlir::Operation*"> {
  let rule = [{
    {{
      Operation* decl = nullptr;
      ::llvm::SmallVector<::xblang::xlg::TemplateParam> params;
      auto block = getInsertionBlock();
    }}
    ( /*parse the template header*/
      "template" "<"
        @Interleave(
          TemplateParamDecl:$param {{ params.push_back(param.get()); }},
          ",")
      ">"
    ) (
      FuncDecl:$fn {{
        decl = fn.get();
      }} |
      StructDecl:$obj {{
        decl = obj.get();
      }}
    ) {{
      $res = makeDeclTemplate(_0.getLoc(), decl, block, params);
    }}
  }];
}

//===----------------------------------------------------------------------===//
// XB parse template parameters
//===----------------------------------------------------------------------===//

def XB_TemplateParamDeclStx :
    Production<"TemplateParamDecl",
               "::xblang::xlg::TemplateParam"> {
  let rule = [{
    /*parse type templates*/
    {{ Value init; }}

    "type" Identifier ( "=" Type:$type {{ init = type.get(); }} )? {{
      $res = xlg::TemplateParam(getLoc(_0),
                                _1.getSpelling(),
                                getConceptClass<xbg::TemplateTypeCep>(),
                                init);
    }}

    | /*parse expression value templates*/
    {{ Value type; Value init; }}

    "let" Identifier (":" Type:$ty {{ type = ty.get(); }})?
      ( "=" Expr:$expr {{ init = expr.get(); }} )? {{
      $res = xlg::TemplateParam(getLoc(_0),
                                _1.getSpelling(),
                                getConceptClass<xbg::TemplateExprCep>(),
                                init);
    }}
  }];
}

//===----------------------------------------------------------------------===//
// XB parse variable declarations
//===----------------------------------------------------------------------===//

def XB_VarDeclStx : Production<"VarDecl", "::mlir::Value"> {
  let rule = XB_VarLikeStx<"VarDecl", "VarDeclCep">.rule;
}

#endif // XBLANG_DECL_SYNTAX
