#ifndef OMP_SYNTAX
#define OMP_SYNTAX

include "xblang/Syntax/SyntaxBase.td"
include "xblang/Syntax/CommonLex.td"
include "xblang/Lang/XBLang/Syntax/Syntax.td"


//===----------------------------------------------------------------------===//
// Parse omp syntax
//===----------------------------------------------------------------------===//

def OMP_StmtStx :
    Production<"Stmt", "::mlir::Operation*"> {
  let rule = [{
    "omp" (
      Parallel:$stmt {{
        $res = stmt.get();
      }}
    )
  }];
}

def OMP_DataSharingStx :
    Production<"DataSharing", "::mlir::SmallVector<DataSharingInfo>"> {
  let rule = [{
    {{ ::mlir::SmallVector<DataSharingInfo> info; }}
    /*parse data sharing clauses*/ (
      /*parse private variables*/
      "private" "("
        @OneOrMore(Identifier:$id {{
            info.push_back(DataSharingInfo::makePrivate(id));
          }}, ",")
        ")" |
      /*parse firstprivate variables*/
      "firstprivate" "("
        @OneOrMore(Identifier:$id {{
            info.push_back(DataSharingInfo::makeFirstPrivate(id));
          }}, ",")
        ")" |
      /*parse shared variables*/
      "shared" "("
        @OneOrMore(Identifier:$id {{
            info.push_back(DataSharingInfo::makeShared(id));
          }}, ",")
        ")"
    )* {{
      $res = info;
    }}
  }];
}

def OMP_ParallelStx :
    Production<"Parallel", "::mlir::Operation*"> {
  let rule = [{{{
      xblang::InsertionBlock block;
    }}
    /*parse the header*/
    "parallel" DataSharing:$dataSharingClauses
    (
      {{ block = builder.getInsertionBlock(); }}
      #dyn("xblang::xlg::Stmt")
    )
    /*create the op*/ {{
      $res = createParallelOp(getLoc(_0), block, dataSharingClauses.get());
    }}
  }];
}

//===----------------------------------------------------------------------===//
// OMP parser
//===----------------------------------------------------------------------===//

def OMPSyntax : Parser<"OMPParser", XBLangLexer,
    [ParserTrait<"XLGBuilderRef", "::xblang::xlg">]> {
  let macros = [
    ParserMacro<"OneOrMore", ["expr", "sep"], [{
        expr (sep expr)*
    }]>,
  ];
  let productions = [
    OMP_StmtStx,
    OMP_DataSharingStx,
    OMP_ParallelStx
  ];
  let startSymbol = "Stmt";
  let defaultToken = "Identifier";
  let cppNamespace = "::omp";
  let extraClassDeclaration = [{
    $parserName(xblang::XBContext *ctx, ::xblang::xlg::XLGBuilder& builder,
                ::xblang::SourceManager& srcManager, Lexer& lexer);
    /// Creates a launch operation.
    mlir::Operation* createParallelOp(mlir::Location loc,
                                      xblang::InsertionBlock& block,
                                      mlir::ArrayRef<DataSharingInfo> sharingClauses);
  }];
}

#endif // OMP_SYNTAX
