#ifndef XBLANG_SYNTAX
#define XBLANG_SYNTAX

include "xblang/Syntax/SyntaxBase.td"
include "xblang/Syntax/CommonLex.td"

//===----------------------------------------------------------------------===//
// XB Lexer
//===----------------------------------------------------------------------===//

// XBLang main automata
def XBLangMainAutomata : FiniteAutomata<"Main"> {
  let loop = 1;
  // Definitions required by the IntLiteral class
  let definitions = [
    DigitStx,
    BinaryDigitStx,
    OctalDigitStx,
    HexDigitStx,
    NonZeroStx,
    DigitSeqStx,
    DigitSeqZStx,
    AlphaStx,
    AlphaNumStx
  ];
  list<Rule> punctuationRules = [
    Literal<"Plus", "+">,
    Literal<"Dash", "-">,
    Literal<"Asterisk", "*">,
    Literal<"Slash", "/">,
    Literal<"Exclamation", "!">,
    Literal<"Ampersand", "&">,
    Literal<"VerBar", "|">,
    Literal<"Hat", "^">,
    Literal<"Less", "<">,
    Literal<"Greater", ">">,
    Literal<"Equal", "=">,
    Literal<"Percent", "%">,
    Literal<"Dot", ".">,
    Literal<"Comma", ",">,
    Literal<"Colon", ":">,
    Literal<"Semicolon", ";">,
    Literal<"Question", "?">,
    Literal<"LBrace", "{">,
    Literal<"RBrace", "}">,
    Literal<"LParen", "(">,
    Literal<"RParen", ")">,
    Literal<"LBracket", "[">,
    Literal<"RBracket", "]">,
    Literal<"Namespace", "::">,
    Literal<"And", "&&">,
    Literal<"Or", "||">,
    Literal<"Equality", "==">,
    Literal<"NEq", "!=">,
    Literal<"LEq", "<=">,
    Literal<"GEq", ">=">,
    Literal<"Spaceship", "<=>">,
    Literal<"LShift", "<<">,
    Literal<"RShift", ">>">,
    Literal<"Increment", "++">,
    Literal<"Decrement", "--">,
    Literal<"CompoundPlus", "+=">,
    Literal<"CompoundMinus", "-=">,
    Literal<"CompoundMultiply", "*=">,
    Literal<"CompoundDivide", "/=">,
    Literal<"CompoundMod", "%=">,
    Literal<"CompoundLShift", "<<=">,
    Literal<"CompoundRShift", ">>=">,
    Literal<"CompoundBAnd", "&=">,
    Literal<"CompoundBOr", "|=">,
    Literal<"CompoundBXor", "^=">,
    Literal<"CompoundAnd", "&&=">,
    Literal<"CompoundOr", "||=">,
    Literal<"LArrow", "<-">,
    Literal<"RArrow", "->">,
  ];
  // Rules in the automata
  let rules = !listconcat(
    [
      StringLiteralStx,
      CommentStx,
      InlineCommentStx,
      IdentifierStx
    ],
    punctuationRules,
    IntAutomata.rules,
    FloatAutomata.rules
  );
}

// XBLang floating-point literal token class
def XB_BoolLiteralStx : TokenClass<"BoolLit"> {
  let tokens = [
    Keyword<"True", "true">,
    Keyword<"False", "false">,
  ];
}

// XBLang lexer
def XBLangLexer : Lexer<"XBLangLexer"> {
  let cppNamespace = "xblang";
  let automatons = [
    XBLangMainAutomata,
    IntAutomata,
    FloatAutomata
  ];
  list<Token> keywords = [
    /*decl and stmt keywords*/
    Keyword<"Import", "import">,
    Keyword<"Let", "let">,
    Keyword<"Struct", "struct">,
    Keyword<"Enum", "enum">,
    Keyword<"Function", "fn">,
    Keyword<"Extern", "extern">,
    Keyword<"Inline", "inline">,
    Keyword<"Noinline", "noinline">,
    Keyword<"In", "in">,
    Keyword<"For", "for">,
    Keyword<"While", "while">,
    Keyword<"Break", "break">,
    Keyword<"Continue", "continue">,
    Keyword<"Return", "return">,
    Keyword<"If", "if">,
    Keyword<"Else", "else">,
    Keyword<"Switch", "switch">,
    Keyword<"Case", "case">,
    Keyword<"Default", "default">,
    Keyword<"Const", "const">,
    Keyword<"Cast", "cast">,
    /*type keywords*/
    Keyword<"Bool", "bool">,
    Keyword<"Int", "int">,
    Keyword<"I8", "i8">,
    Keyword<"I16", "i16">,
    Keyword<"I32", "i32">,
    Keyword<"I64", "i64">,
    Keyword<"UInt", "uint">,
    Keyword<"U8", "u8">,
    Keyword<"U16", "u16">,
    Keyword<"U32", "u32">,
    Keyword<"U64", "u64">,
    Keyword<"Float", "float">,
    Keyword<"F8", "f8">,
    Keyword<"F16", "f16">,
    Keyword<"F32", "f32">,
    Keyword<"F64", "f64">,
    Keyword<"F128", "f128">,
    Keyword<"Auto", "auto">,
    Keyword<"Address", "address">,
    Keyword<"Void", "void">,
    /*literal keywords*/
    Keyword<"True", "true">,
    Keyword<"False", "false">,
    Keyword<"Null", "null">,
    Keyword<"This", "this">,
  ];
  let tokens = !listconcat(
    keywords,
    XBLangMainAutomata.rules,
    [Token<"DialectKeyword">]
  );
  let tokenClasses = [
    IntegerLiteral,
    SignedIntegerLiteral,
    UnsignedIntegerLiteral,
    FloatLiteral,
    XB_BoolLiteralStx
  ];
  let extraClassDeclaration = [{
    static xblang::BinaryOperator toBinaryOp(TokenID kind);
    static xblang::UnaryOperator toUnaryOp(TokenID kind);
    struct IntLiteralInfo {
      using Signedness = mlir::IntegerType::SignednessSemantics;
      int radix = 10;
      unsigned width = 0;
      Signedness signedness = Signedness::Signless;
    };
    static IntLiteralInfo getIntInfo(TokenID kind);
    TokenID parseString(SourceState& state, llvm::StringRef& spelling) const;
    private:
    mutable llvm::SmallVector<std::string> stringLiterals;
  }];
}

//===----------------------------------------------------------------------===//
// XB parser
//===----------------------------------------------------------------------===//

include "xblang/Lang/XBLang/Syntax/Decl.td"
include "xblang/Lang/XBLang/Syntax/Stmt.td"
include "xblang/Lang/XBLang/Syntax/Expr.td"
include "xblang/Lang/XBLang/Syntax/Type.td"

def XBLangSyntax : Parser<"XBLangParser", XBLangLexer, [
      ParserTrait<"XLGBuilder", "::xblang::xlg">
    ]> {
  let macros = [
    ParserMacro<"Interleave", ["expr", "sep"], [{
        expr (sep expr)* | eps
    }]>,
  ];
  let productions = [
    //===------------------------------------------------------------------===//
    // Decl productions
    //===------------------------------------------------------------------===//
    XB_TopDeclStx,
    XB_FuncDeclStx,
    XB_FuncDefStx,
    XB_FuncAttrsStx,
    XB_MemberDeclStx,
    XB_ParamDeclStx,
    XB_StructDeclStx,
    XB_TemplateDeclStx,
    XB_TemplateParamDeclStx,
    XB_VarDeclStx,
    //===------------------------------------------------------------------===//
    // Type productions
    //===------------------------------------------------------------------===//
    XB_TypeStx,
    XB_UnqualifiedTypeStx,
    XB_BuiltinTypeStx,
    XB_IntegerStx,
    XB_FloatStx,
    XB_QualifiedIdentifierTypeStx,
    //===------------------------------------------------------------------===//
    // Stmt productions
    //===------------------------------------------------------------------===//
    XB_StmtStx,
    XB_BreakStmtStx,
    XB_CompoundStmtStx,
    XB_ContinueStmtStx,
    XB_DeclStmtStx,
    XB_ForStmtStx,
    XB_IfStmtStx,
    XB_ReturnStmtStx,
    XB_WhileStmtStx,
    //===------------------------------------------------------------------===//
    // Expr productions
    //===------------------------------------------------------------------===//
    XB_ExprStx,
    XB_BinaryExprStx,
    XB_TopExprStx,
    XB_CoreExprStx,
    XB_ConstExprStx,
    XB_CommaListExprStx,
    XB_RangeExprStx,
    XB_QualifiedIdentifierExprStx
  ];
  let startSymbol = "TopDecl";
  let defaultToken = "Identifier";
  let cppNamespace = "xblang";
  let extraClassDeclaration = [{
    $parserName(XBContext *ctx, ::mlir::Block* block, ::xblang::SourceManager& srcManager, Lexer& lexer);

    /// Parse a source state using the start symbol.
    ::xblang::syntax::ParsingStatus parseState(const SourceState& state);

    /// Returns an integer expression
    mlir::Value getIntExpr(const Token& tok);

    /// Returns a float expression
    mlir::Value getFloatExpr(const Token& tok);

    /// Returns a bool expression
    mlir::Value getBoolExpr(const Token& tok);

    /// Returns a string expression
    mlir::Value getStringExpr(const Token& tok);

    /// Returns a null expression
    mlir::Value getNullExpr();

    struct RangeLoopInfo {
      Token id;
      Value type;
      Value range;
    };
    /// Creates a range for statement
    mlir::Operation* createRangeForStmt(Location loc,
                                        InsertionBlock &&block,
                                        SmallVector<RangeLoopInfo>& ranges);
    /// Builds a template declaration.
    xlg::TemplateOp makeDeclTemplate(const SourceLocation& loc,
                                     Operation* decl,
                                     InsertionBlock& block,
                                     ArrayRef<xlg::TemplateParam> parameters);
  }];
}

#endif // XBLANG_SYNTAX
