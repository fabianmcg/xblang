#ifndef XBLANG_DIALECT_XBLANG_IR_STMT_TD
#define XBLANG_DIALECT_XBLANG_IR_STMT_TD

include "xblang/Dialect/XBLang/IR/Dialect.td"
include "xblang/Dialect/XBLang/IR/Enums.td"
include "xblang/Dialect/XBLang/IR/Type.td"
include "xblang/Dialect/XBLang/IR/Interfaces.td"

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


def IfOp : XBLang_Op<"if", [
    RecursivelySpeculatable,
    AutomaticAllocationScope,
    NoRegionArguments,
    ImplicitCastTrait,
    AffineScope]> {
  let summary = "If Op";
  let description = [{}];
  let arguments = (ins AnyType:$condition);
  let regions = (region AnyRegion:$thenRegion, AnyRegion:$elseRegion);
  let assemblyFormat = "$condition type($condition) attr-dict `:` $thenRegion (`else` $elseRegion^)?";
  let hasVerifier = 1;
}

def LoopOp : XBLang_Op<"loop", [
    RecursiveMemoryEffects,
    DeclareOpInterfaceMethods<LoopLikeOpInterface>,
    DeclareOpInterfaceMethods<RegionBranchOpInterface>,
    RecursivelySpeculatable,
    AutomaticAllocationScope,
    AffineScope]> {
  let summary = "Loop Op";
  let description = [{}];
  let arguments = (ins);
  let regions = (region AnyRegion:$condition,
                        AnyRegion:$body,
                        AnyRegion:$iteration);
  let assemblyFormat = [{
    attr-dict-with-keyword
        `condition` `:` $condition
        `body` `:`  $body
        `iteration` `:`  $iteration}];
  let hasVerifier = 0;
}

def RangeForOp : XBLang_Op<"range_for", [
       AttrSizedOperandSegments,
       RecursiveMemoryEffects,
       DeclareOpInterfaceMethods<LoopLikeOpInterface>,
       DeclareOpInterfaceMethods<RegionBranchOpInterface>,
       RecursivelySpeculatable,
       AutomaticAllocationScope]> {
  let summary = "For Op";
  let description = [{}];
  let arguments = (ins Variadic<AnyType>:$variables, Variadic<RangeType>:$ranges, UnitAttr:$affine);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "`(`custom<RangeList>($variables, type($variables), $ranges, type($ranges)) `)` attr-dict-with-keyword $body";
  let hasVerifier = 0;
  let builders = [
    OpBuilder<(ins "Value":$variable, "Value":$range, CArg<"bool", "false">:$affine), [{
      build($_builder, $_state, ValueRange({variable}), ValueRange({range}), affine);
    }]>,
  ];
  let extraClassDeclaration = [{    
    Value getRange(); 
    Value getVariable();
  }];
  let extraClassDefinition = [{
    Value RangeForOp::getRange() {
      auto ranges = getRanges();
      if (ranges.size())
        return ranges[0];
      return nullptr;
    }
    Value RangeForOp::getVariable() {
      auto vars = getVariables();
      if (vars.size())
        return vars[0];
      return nullptr;
    }
  }];
}

//===----------------------------------------------------------------------===//
// XB return Op
//===----------------------------------------------------------------------===//

def ReturnOp : XBLang_Op<"return", [
      Pure,
      MemRefsNormalizable,
      ReturnLike,
      Terminator,
      ImplicitCastTrait,
      DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface>
    ]> {
  let summary = "Return from function";
  let description = [{}];
  let arguments = (ins Variadic<AnyType>:$input, OptionalAttr<TypeAttr>:$type);
  let assemblyFormat = "($input^ `:` type($input))? attr-dict";
  let builders = [
    OpBuilder<(ins CArg<"ValueRange", "std::nullopt">:$value,
                    CArg<"Type", "nullptr">:$type), [{
      TypeAttr typeAttr = type ? TypeAttr::get(type) : nullptr;
      build($_builder, $_state, value, typeAttr);
    }]>
  ];
  let extraClassDeclaration = [{
    bool hasOperands() { return getNumOperands() != 0; }
  }];
  let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// XB scope Op
//===----------------------------------------------------------------------===//

def ScopeOp : XBLang_Op<"scope", [
      AffineScope,
      AutomaticAllocationScope,
      NoRegionArguments
    ]> {
  let regions = (region AnyRegion:$body);
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins CArg<"bool", "true">:$addBlock), [{
    Region *bodyRegion = $_state.addRegion();
    if (addBlock)
      bodyRegion->push_back(new Block());
  }]>];
  let extraClassDeclaration = [{
    Block *getFrontBlock();
    Block *getBackBlock();
  }];
  let assemblyFormat = [{
  	custom<RegionWithImplicitYield>($body) attr-dict-with-keyword
  }];
  let hasCanonicalizeMethod = 1;
}

def YieldOp : XBLang_Op<"yield", [Pure,
                                 Terminator,
                                 DeclareOpInterfaceMethods<
                                   RegionBranchTerminatorOpInterface>
                                 ]> {
  let summary = "Return control from a scope";
  let description = [{}];
  let arguments = (ins YieldKind:$kind, Variadic<AnyType>:$arguments);
  let assemblyFormat = "$kind attr-dict ($arguments^ type($arguments))?";
  let hasVerifier = 0;
}
#endif
