#ifndef XBLANG_DIALECT_PARALLEL_IR_ENUMS_TD
#define XBLANG_DIALECT_PARALLEL_IR_ENUMS_TD

include "mlir/IR/EnumAttr.td"

def DataSharingKindEnum: I32EnumAttr<"DataSharingKind", "Data Sharing kind", [
    I32EnumAttrCase<"Private", 1>,
    I32EnumAttrCase<"Firstprivate", 2>,
    I32EnumAttrCase<"Shared", 3>
  ]> {
  let cppNamespace = "::mlir::par";
  let stringToSymbolFnName = "stringToDataSharingKind";
  let symbolToStringFnName = "toString";
}

def MapKindEnum: I32EnumAttr<"MapKind", "Map kind", [
    I32EnumAttrCase<"present", 0>,
    I32EnumAttrCase<"to", 1>,
    I32EnumAttrCase<"from", 2>,
    I32EnumAttrCase<"toFrom", 3>,
    I32EnumAttrCase<"allocate", 4>,
    I32EnumAttrCase<"destroy", 5>,
  ]> {
  let cppNamespace = "::mlir::par";
  let stringToSymbolFnName = "stringToMapKind";
  let symbolToStringFnName = "toString";
}

def ParallelHierarchyEnum: I32EnumAttr<"ParallelHierarchy", "Parallel Hierarchy", [
    I32EnumAttrCase<"automatic", 0>,
    I32EnumAttrCase<"scalar", 1>,
    I32EnumAttrCase<"vector", 2>,
    I32EnumAttrCase<"matrix", 4>,
    I32EnumAttrCase<"tensor", 8>,
    I32EnumAttrCase<"v2s", 3>,
    I32EnumAttrCase<"m2s", 5>,
    I32EnumAttrCase<"m2v", 6>,
    I32EnumAttrCase<"t2s", 9>,
    I32EnumAttrCase<"t2v", 10>,
    I32EnumAttrCase<"t2m", 12>,
  ]> {
  let cppNamespace = "::mlir::par";
  let stringToSymbolFnName = "stringToParallelHierarchy";
  let symbolToStringFnName = "toString";
}

def ReduceOpsEnum: I32EnumAttr<"ReduceOps", "Reduce Ops", [
    I32EnumAttrCase<"Add", 0, "+">,
    I32EnumAttrCase<"Mul", 1, "*">,
    I32EnumAttrCase<"And", 2, "&&">,
    I32EnumAttrCase<"Or", 3, "||">,
    I32EnumAttrCase<"Xor", 4, "^">,
    I32EnumAttrCase<"Max", 5, "max">,
    I32EnumAttrCase<"Min", 6, "min">
  ]> {
  let cppNamespace = "::mlir::par";
  let stringToSymbolFnName = "stringToReduceOps";
  let symbolToStringFnName = "toString";
}

def AtomicOpsEnum: I32EnumAttr<"AtomicOps", "Atomic Ops", [
    I32EnumAttrCase<"Add", 0, "+">,
    I32EnumAttrCase<"Mul", 1, "*">,
    I32EnumAttrCase<"And", 2, "&&">,
    I32EnumAttrCase<"Or", 3, "||">,
    I32EnumAttrCase<"Xor", 4, "^">,
    I32EnumAttrCase<"Assign", 5, "=">,
    I32EnumAttrCase<"Max", 6, "max">,
    I32EnumAttrCase<"Min", 7, "min">
  ]> {
  let cppNamespace = "::mlir::par";
  let stringToSymbolFnName = "stringToAtomicOps";
  let symbolToStringFnName = "toString";
}

def Par_LoopComparatorOp: I32EnumAttr<"LoopComparatorOp", "Loop comparator Op", [
    I32EnumAttrCase<"None", 0, "none">,
    I32EnumAttrCase<"Less", 1, "<">,
    I32EnumAttrCase<"Greater", 2, ">">,
    I32EnumAttrCase<"LEQ", 3, "<=">,
    I32EnumAttrCase<"GEQ", 4, ">=">
  ]> {
  let cppNamespace = "::mlir::par";
  let stringToSymbolFnName = "stringToLoopComparator";
  let symbolToStringFnName = "toString";
}

def Par_LoopStepOp: I32EnumAttr<"LoopStepOp", "Loop step Op", [
    I32EnumAttrCase<"None", 0, "none">,
    I32EnumAttrCase<"Add", 1, "+">,
    I32EnumAttrCase<"Sub", 2, "-">,
    I32EnumAttrCase<"Mul", 3, "*">,
    I32EnumAttrCase<"Div", 4, "/">,
    I32EnumAttrCase<"LShift", 5, "<<">,
    I32EnumAttrCase<"RShift", 6, ">>">
  ]> {
  let cppNamespace = "::mlir::par";
  let stringToSymbolFnName = "stringToLoopStep";
  let symbolToStringFnName = "toString";
}

#endif
