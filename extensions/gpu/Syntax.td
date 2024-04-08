#ifndef GPU_SYNTAX
#define GPU_SYNTAX

include "xblang/Syntax/SyntaxBase.td"
include "xblang/Syntax/CommonLex.td"
include "xblang/Lang/XBLang/Syntax/Syntax.td"


//===----------------------------------------------------------------------===//
// Parse gpu syntax
//===----------------------------------------------------------------------===//

def GPU_StmtStx :
    Production<"Stmt", "::mlir::Operation*"> {
  let rule = [{
    "gpu" "::"  (
      Region:$stmt {{
        $res = stmt.get();
      }}
    )
  }];
}

def GPU_RegionStx :
    Production<"Region", "::mlir::Operation*"> {
  let rule = [{{{
      mlir::SmallVector<mlir::Value, 3> bsz, gsz;
      xblang::InsertionBlock block;
      registerRegionCombinators(*this);
    }}
    /*parse the header*/
    "region" "<<"
      "[" @Dim3(#dyn("xblang::xlg::Expr"):$bv {{ bsz.push_back(bv.get()); }}) "]"
      ","
      "[" @Dim3(#dyn("xblang::xlg::Expr"):$gv {{ gsz.push_back(gv.get()); }}) "]"
    ">>" (
      {{ block = builder.getInsertionBlock(); }}
      #dyn("xblang::xlg::Stmt")
    )

    /*create the op*/ {{
      bsz.resize(3);
      gsz.resize(3);
      $res = createLaunchOp(getLoc(_0), bsz, gsz, block);
    }}
  }];
}

def GPU_IdStx :
    Production<"Id", "::mlir::Value"> {
  let rule = [{
    {{ ::mlir::gpu::Dimension dim; int kind = -1; auto tok = _tok; }}
    (
      "threadIdx" {{
        kind = 0;
      }} |
      "blockIdx" {{
        kind = 1;
      }} |
      "blockDim" {{
        kind = 2;
      }} |
      "gridDim" {{
        kind = 3;
      }}
    ) "." (
      "x" {{
        dim = ::mlir::gpu::Dimension::x;
      }} |
      "y" {{
        dim = ::mlir::gpu::Dimension::y;
      }} |
      "z" {{
        dim = ::mlir::gpu::Dimension::z;
      }}
    ) {{
      $res = createIdDimOp(getLoc(tok), kind, dim);
    }}
  }];
}

//===----------------------------------------------------------------------===//
// GPU parser
//===----------------------------------------------------------------------===//

def GPUSyntax : Parser<"GPUParser", XBLangLexer,
    [ParserTrait<"XLGBuilderRef", "::xblang::xlg">]> {
  let macros = [
    ParserMacro<"Dim3", ["expr"], [{
       expr ("," expr ("," expr)? )?
    }]>,
  ];
  let productions = [
    GPU_StmtStx,
    GPU_RegionStx,
    GPU_IdStx
  ];
  let startSymbol = "Stmt";
  let defaultToken = "Identifier";
  let cppNamespace = "::gpu";
  let extraClassDeclaration = [{
    $parserName(xblang::XBContext *ctx, ::xblang::xlg::XLGBuilder& builder,
                ::xblang::SourceManager& srcManager, Lexer& lexer);
    /// Creates a launch operation.
    mlir::Operation* createLaunchOp(mlir::Location loc,
                                    mlir::SmallVectorImpl<mlir::Value>& bsz,
                                    mlir::SmallVectorImpl<mlir::Value>& gsz,
                                    xblang::InsertionBlock& block);
    /// Creates an id or dim operation.
    mlir::Value createIdDimOp(mlir::Location loc, int kind, mlir::gpu::Dimension);
  }];
}

#endif // GPU_SYNTAX
