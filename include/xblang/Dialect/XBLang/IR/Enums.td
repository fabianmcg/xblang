#ifndef XBLANG_DIALECT_XBLANG_IR_ENUMS_TD
#define XBLANG_DIALECT_XBLANG_IR_ENUMS_TD

include "mlir/IR/EnumAttr.td"

/******************************************************************************
* Decl enumerations
******************************************************************************/

def VarKindEnum: I32EnumAttr<"VarKind", "Variable kind", [
  I32EnumAttrCase<"local", 0>,
  I32EnumAttrCase<"param", 1>,
  I32EnumAttrCase<"global", 2>
  ]> {
  let cppNamespace = "::xblang::xb";
  let stringToSymbolFnName = "stringToVarKind";
  let symbolToStringFnName = "toString";
}

/******************************************************************************
* Expr enumerations
******************************************************************************/

class OperatorCase<string i, string r = i, bit c = 0> {
  string id = i;
  string repr = r;
  bit isCompound = c;
}

class Operator<OperatorCase op, int i> {
  OperatorCase case = op;
  int counter = i;
}

def Operators {
  list<OperatorCase> OperatorsTmp = [
  OperatorCase<"Add", "+", 1>,
  OperatorCase<"Sub", "-", 1>,
  OperatorCase<"Mul", "*", 1>,
  OperatorCase<"Div", "/", 1>,
  OperatorCase<"Mod", "%", 1>,
  OperatorCase<"Pow", "**", 1>,
  OperatorCase<"LShift", "<<", 1>,
  OperatorCase<"RShift", ">>", 1>,
  OperatorCase<"BinaryAnd", "&", 1>,
  OperatorCase<"BinaryOr", "|", 1>,
  OperatorCase<"BinaryXor", "^", 1>,
  OperatorCase<"And", "&&", 1>,
  OperatorCase<"Or", "||", 1>,
  OperatorCase<"Equal", "==">,
  OperatorCase<"NEQ", "!=">,
  OperatorCase<"Less", "<">,
  OperatorCase<"Greater", ">">,
  OperatorCase<"LEQ", "<=">,
  OperatorCase<"GEQ", ">=">,
  OperatorCase<"Spaceship", "<=>">,
  OperatorCase<"At", "@">,
  OperatorCase<"Dot", ".">,
  OperatorCase<"Comma", ",">,
  OperatorCase<"Colon", ":">,
  OperatorCase<"Ternary", "?">,
  OperatorCase<"Ellipsis", "...">,
  OperatorCase<"Assign", "=">,
  OperatorCase<"lastBinOp">,
  ];
  list<Operator> baseCases = !foldl(
    [Operator<OperatorCase<"firstBinOp">, 0>],
    OperatorsTmp,
    result,
    tmp, 
    !listconcat(result,
      [Operator<tmp, !add(!foldl(0, result, acc, t, t.counter), 1)>]
    )
  );
  list<Operator> compoundCases = !filter(op, baseCases, op.case.isCompound);
  list<I32EnumAttrCase> cases = !listconcat(
    !foreach(op,
             baseCases,
             I32EnumAttrCase<op.case.id, op.counter, op.case.repr>),
             [I32EnumAttrCase<"Compound", 64>],
             !foreach(op,
                      compoundCases,
                      I32EnumAttrCase<!strconcat("Compound", op.case.id), 
                                      !or(op.counter, 64),
                                      !strconcat(op.case.repr, "=")>
             )
  );
 }

def BinaryOperator: I32EnumAttr<"BinaryOperator", "Binary op kind", Operators.cases> {
  let cppNamespace = "::xblang";
  let stringToSymbolFnName = "stringToBinarayOpKind";
  let symbolToStringFnName = "toString";
}

def UnaryOperator: I32EnumAttr<"UnaryOperator", "Unary op kind", [
  I32EnumAttrCase<"None", 0>,
  I32EnumAttrCase<"Address", 1, "&">,
  I32EnumAttrCase<"Dereference", 2, "*">,
  I32EnumAttrCase<"Plus", 3, "+">,
  I32EnumAttrCase<"Minus", 4, "-">,
  I32EnumAttrCase<"Negation", 5, "!">,
  I32EnumAttrCase<"Increment", 6, "++">,
  I32EnumAttrCase<"Decrement", 7, "--">,
  I32EnumAttrCase<"PostIncrement", 8, "^++">,
  I32EnumAttrCase<"PostDecrement", 9, "^--">,
  ]> {
  let cppNamespace = "::xblang";
  let stringToSymbolFnName = "stringToUnarayOpKind";
  let symbolToStringFnName = "toString";
}

/******************************************************************************
* Stmt enumerations
******************************************************************************/

def YieldKind: I32EnumAttr<"YieldKind", "Variable kind", [
  I32EnumAttrCase<"Fallthrough", 0>,
  I32EnumAttrCase<"Continue", 1>,
  I32EnumAttrCase<"Break", 2>,
  ]> {
  let cppNamespace = "::xblang::xb";
  let stringToSymbolFnName = "stringToYieldKind";
  let symbolToStringFnName = "toString";
}

#endif
