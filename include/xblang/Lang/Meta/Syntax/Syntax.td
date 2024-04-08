#ifndef META_SYNTAX
#define META_SYNTAX

include "xblang/Syntax/SyntaxBase.td"
include "xblang/Syntax/CommonLex.td"
include "xblang/Lang/XBLang/Syntax/Syntax.td"


//===----------------------------------------------------------------------===//
// Parse meta syntax
//===----------------------------------------------------------------------===//

def Meta_AttrStx :
    Production<"Attr", "::mlir::Attribute"> {
  let rule = [{
    "mlir" "::" "attr" "(" StringLiteral:$literal ")" {{
      $res = getAttr(_0.getLoc(), literal.getSpelling());
    }}
  }];
}

def Meta_InlineStx :
    Production<"Inline", "::mlir::Value"> {
  let rule = [{ {{
      SmallVector<std::pair<Token, StringRef>> args;
      std::pair<StringRef, StringRef> ret{};
    }}
    "mlir" "::" "inline"

    /*parse the arguments*/ ("("
      @Interleave(Identifier:$id ":" StringLiteral:$ty {{
        args.push_back({id, ty.getSpelling()});
      }}, ",")
    ")")?

    /*parse the return*/ (
      ("->" Identifier:$id1 ("::" Identifier:$id2 {{
        ret.second = id2.getSpelling();
      }})? {{
        if (ret.second.empty())
          ret.second = id1.getSpelling();
        else
          ret.first = id1.getSpelling();
      }})
    )?
    StringLiteral:$literal {{
      auto val = getInlineValue(_0.getLoc(), args, ret, literal.getSpelling());
      if (!val)
        $assertError
      $res = val;
    }}
  }];
}

def Meta_TopStx :
    Production<"Top", "::mlir::Operation*"> {
  let rule = [{
    "mlir" "::" "top"
    StringLiteral:$literal {{
      auto status = getInlineOp(_0.getLoc(), literal.getSpelling());
      if (!status.isSuccess())
        $assertError
      $res = nullptr;
    }}
  }];
}

def Meta_TypeStx :
    Production<"Type", "::mlir::Value"> {
  let rule = [{
    "mlir" "::" "type" "(" StringLiteral:$literal ")" {{
      $res = getType(getLoc(_0), literal.getSpelling());
    }}
  }];
}

//===----------------------------------------------------------------------===//
// Meta parser
//===----------------------------------------------------------------------===//

def MetaSyntax : Parser<"MetaParser", XBLangLexer,
    [ParserTrait<"XLGBuilderRef", "::xblang::xlg">]> {
  let macros = [
    ParserMacro<"Interleave", ["expr", "sep"], [{
        expr (sep expr)* | eps
    }]>,
  ];
  let productions = [
    Meta_AttrStx,
    Meta_InlineStx,
    Meta_TopStx,
    Meta_TypeStx,
  ];
  let startSymbol = "Decl";
  let defaultToken = "Identifier";
  let cppNamespace = "::xblang::meta";
  let extraClassDeclaration = [{
    $parserName(XBContext *ctx, ::xblang::xlg::XLGBuilder& builder,
                ::xblang::SourceManager& srcManager, Lexer& lexer);
    ::mlir::Attribute getAttr(const SourceLocation& loc, StringRef literal);
    syntax::PureParsingStatus getInlineOp(const SourceLocation& loc, StringRef literal);
    ::mlir::Value getInlineValue(const SourceLocation& loc,
                                 ArrayRef<std::pair<Token, StringRef>> args,
                                 std::pair<StringRef, StringRef> ret,
                                 StringRef literal);
    ::mlir::Value getType(Location loc, StringRef literal);
  }];
}

#endif // META_SYNTAX
