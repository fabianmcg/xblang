#ifndef XBLANG_DIALECT_XBLANG_IR_EXPR_TD
#define XBLANG_DIALECT_XBLANG_IR_EXPR_TD

include "xblang/Dialect/XBLang/IR/Dialect.td"
include "xblang/Dialect/XBLang/IR/Enums.td"
include "xblang/Dialect/XBLang/IR/Type.td"
include "xblang/Dialect/XBLang/IR/Interfaces.td"

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def ArrayOp : XBLang_Op<"array"> {
  let summary = "Array op";
  let description = [{
  }];
  let arguments = (ins AnyType:$base, Variadic<AnyType>:$index);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $base type($base) ` ` `[` $index type($index) `]` `:` type($result) attr-dict
  }];
}

def ArrayViewOp : XBLang_Op<"array_view"> {
  let summary = "Array view op";
  let description = [{
  }];
  let arguments = (ins AnyType:$base, Variadic<RangeType>:$ranges);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $base type($base) ` ` `[` $ranges type($ranges) `]` `:` type($result) attr-dict
  }];
}

def BinaryOp : XBLang_Op<"bop"> {
  let summary = "Binary op";
  let description = [{
  }];
  let arguments = (ins BinaryOperator:$op, AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $op $lhs type($lhs) `,` $rhs type($rhs) `:` type($result) attr-dict
  }];
}

def CallOp : XBLang_Op<"call", [DeclareOpInterfaceMethods<CallOpInterface>]> {
  let summary = "Call op";
  let description = [{
  }];
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$result);
  let extraClassDeclaration = [{
    FunctionType getCalleeType();
  }];
  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// XB cast Op
//===----------------------------------------------------------------------===//

def CastOp : XBLang_Op<"cast", [
    DeclareOpInterfaceMethods<ConditionallySpeculatable, ["getSpeculatability"]>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface, ["getEffects"]>
  ]> {
  let summary = "Cast op";
  let description = [{
  }];
  let arguments = (ins UnitAttr:$low, UnitAttr:$unknown, AnyType:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $value `:` type($value) `->` type($result) attr-dict
  }];
  let builders = [
    OpBuilder<(ins "Type":$type, "Value":$value, CArg<"bool", "false">:$low, CArg<"bool", "false">:$unk), [{
      build($_builder, $_state, type, low, unk, value);
    }]>
  ];
  let hasFolder = 1;
}

def CollapseMemRefOp : XBLang_Op<"collapse_memref"> {
  let summary = "Collapse MemRef op";
  let description = [{
  }];
  let arguments = (ins AnyMemRef:$input);
  let results = (outs AnyMemRef:$result);
  let assemblyFormat = [{
    $input type($input) `:` type($result) attr-dict
  }];
}

def ConstantOp : XBLang_Op<"constant", [ConstantLike,
                                       Pure]> {
  let summary = "Defines a constant";
  let description = [{
  }];
  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    `(` custom<Constant>($value) `)` `:` type($result) attr-dict
  }];
  let builders = [
    OpBuilder<(ins "TypedAttr":$value), [{
      assert(value);
      build($_builder, $_state, value.getType(), value);
    }]>,
  ];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// XB from_elements Op
//===----------------------------------------------------------------------===//

def FromElementsOp : XBLang_Op<"from_elements"> {
  let summary = "To tensor op";
  let description = [{
  }];
  let arguments = (ins Variadic<AnyType>:$elements);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $elements type($elements) `:` type($result) attr-dict
  }];
}

def GetElementOp : XBLang_Op<"get_member"> {
  let summary = "Get member op";
  let description = [{
  }];
  let arguments = (ins AnyType:$base, TypedAttrInterface:$index);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $base type($base) ` ` `[` $index `]` `:` type($result) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// XB load Op
//===----------------------------------------------------------------------===//

def LoadOp : XBLang_Op<"load"> {
  let summary = "Load op";
  let description = [{
  }];
  let arguments = (ins ReferenceType:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $value `:` type($value) `->` type($result) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// XB nullptr Op
//===----------------------------------------------------------------------===//

def NullPtrOp : XBLang_Op<"nullptr", [Pure, ConstantLike]> {
  let summary = "Nullptr";
  let description = [{}];
  let results = (outs AnyTypeOf<[AddressType, PointerType]>:$result);
  let assemblyFormat = "type($result) attr-dict";
  let hasFolder = 1;
  let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// XB range Op
//===----------------------------------------------------------------------===//

def RangeOp : XBLang_Op<"range", [Pure, ImplicitCastTrait]> {
  let summary = "Range op";
  let description = [{
  }];
  let arguments = (ins BinaryOperator:$comparator,
                       AnyType:$begin,
                       AnyType:$end,
                       OptionalAttr<BinaryOperator>:$stepOp,
                       Optional<AnyType>:$step);
  let results = (outs RangeType:$result);
  let assemblyFormat = [{
    $begin type($begin) $comparator $end type($end) (`step` $stepOp $step type($step)^)? `:` type($result) attr-dict
  }];
}

def RefOp : XBLang_Op<"ref"> {
  let summary = "References a symbol";
  let description = [{
  }];
  let arguments = (ins FlatSymbolRefAttr:$identifier);
  let results = (outs AnyType:$reference);
  let assemblyFormat = [{
    $identifier `:` type($reference) attr-dict
  }];
}

def ReinterpretCastOp : XBLang_Op<"reinterpret_cast"> {
  let summary = "Converts a value to any other type";
  let description = [{
  }];
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $input `:` type($input) `->` type($result) attr-dict
  }];
}

def SelectOp : XBLang_Op<"select"> {
  let summary = "Select op";
  let description = [{
  }];
  let arguments = (ins AnyType:$condition, AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $condition type($condition) `?` $lhs type($lhs) `,` $rhs type($rhs) `:` type($result) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// XB sizeof Op
//===----------------------------------------------------------------------===//

def SizeOfOp : XBLang_Op<"sizeof", [Pure,
                                   ConstantLike,
                                  ]> {
  let summary = "Size of a type";
  let description = [{}];
  let arguments = (ins TypeAttr:$type);
  let results = (outs Index:$result);
  let assemblyFormat = "$type attr-dict";
  let hasFolder = 1;
  let hasVerifier = 0;
}

def UnaryOp : XBLang_Op<"uop"> {
  let summary = "Unary op";
  let description = [{
  }];
  let arguments = (ins UnaryOperator:$op, AnyType:$expr);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $op $expr type($expr) `:` type($result) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// XB store Op
//===----------------------------------------------------------------------===//

def StoreOp : XBLang_Op<"store"> {
  let summary = "Store op";
  let description = [{
  }];
  let arguments = (ins ReferenceType:$address, AnyType:$value);
  let assemblyFormat = [{
    $address `:` type($address) `,` $value `:` type($value) attr-dict
  }];
}

#endif
