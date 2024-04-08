#ifndef XBLANG_DIALECT_XBLANG_IR_TYPE_TD
#define XBLANG_DIALECT_XBLANG_IR_TYPE_TD

include "xblang/Dialect/XBLang/IR/Dialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// XB address type
//===----------------------------------------------------------------------===//

def AddressType : XBLang_Type<"Address", "address", [
    MemRefElementTypeInterface
  ]> {
  let summary = "XB address type";
  let parameters = (ins OptionalParameter<"Attribute">:$memorySpace);
  let builders = [
    TypeBuilder<(ins
      CArg<"Attribute", "{}">:$memorySpace), [{
        return Base::get($_ctxt, memorySpace);
      }]>
  ];
  let assemblyFormat = "(`<` $memorySpace^ `>`)?";
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// XB generic type
//===----------------------------------------------------------------------===//

def GenericType : XBLang_Type<"Any", "any"> {
  let summary = "XB any type";
  let assemblyFormat = "";
  let genVerifyDecl = 0;
}

//===----------------------------------------------------------------------===//
// XB pointer type
//===----------------------------------------------------------------------===//

def PointerType : XBLang_Type<"Pointer", "ptr", [MemRefElementTypeInterface]> {
  let summary = "XB pointer type";
  let description = [{
    The `xb.ptr` type represents typed pointers.
  }];
  let parameters = (ins "mlir::Type":$pointee,
                        OptionalParameter<"Attribute">:$memorySpace);
  let builders = [
    TypeBuilder<(ins
      "::mlir::Type":$pointee,
      CArg<"Attribute", "{}">:$memorySpace), [{
        return Base::get($_ctxt, pointee, memorySpace);
      }]>,
    TypeBuilderWithInferredContext<(ins
      "::mlir::Type":$pointee,
      CArg<"Attribute", "{}">:$memorySpace), [{
        return Base::get(pointee.getContext(), pointee, memorySpace);
      }]>
  ];
  let assemblyFormat = "`<` $pointee (`,` $memorySpace^)? `>`";
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// XB range type
//===----------------------------------------------------------------------===//

def RangeType : XBLang_Type<"Range", "range_t"> {
  let summary = "XB range type";
  let parameters = (ins "mlir::Type":$iteratorType);
  let assemblyFormat = "`<` $iteratorType `>`";
}

//===----------------------------------------------------------------------===//
// XB reference type
//===----------------------------------------------------------------------===//

def ReferenceType : XBLang_Type<"Reference", "ref", [
    MemRefElementTypeInterface
  ]> {
  let summary = "XB reference type";
  let description = [{
    The `xb.ref` type represent value references.
  }];
  let parameters = (ins "mlir::Type":$pointee,
                        OptionalParameter<"Attribute">:$memorySpace);
  let builders = [
    TypeBuilder<(ins
      "::mlir::Type":$pointee,
      CArg<"Attribute", "{}">:$memorySpace), [{
        if (auto type = mlir::dyn_cast<ReferenceType>(pointee))
          return type;
        return Base::get($_ctxt, pointee, memorySpace);
      }]>,
    TypeBuilderWithInferredContext<(ins
      "::mlir::Type":$pointee,
      CArg<"Attribute", "{}">:$memorySpace), [{
        if (auto type = mlir::dyn_cast<ReferenceType>(pointee))
          return type;
        return Base::get(pointee.getContext(), pointee, memorySpace);
      }]>
  ];
  let assemblyFormat = "`<` $pointee (`,` $memorySpace^)? `>`";
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// XB string type
//===----------------------------------------------------------------------===//

def StringType : XBLang_Type<"String", "string", [MemRefElementTypeInterface]> {
  let summary = "XB string type";
  let assemblyFormat = "";
  let genVerifyDecl = 0;
}

//===----------------------------------------------------------------------===//
// XB struct type
//===----------------------------------------------------------------------===//

def StructType : XBLang_Type<"Struct", "struct", [MemRefElementTypeInterface]> {
  let summary = "XB struct type";
  let description = [{
  }];
  let parameters = (ins ArrayRefParameter<"mlir::Type", "members">:$members);
  let assemblyFormat = "`<` $members `>`";
}

//===----------------------------------------------------------------------===//
// XB void type
//===----------------------------------------------------------------------===//

def VoidType : XBLang_Type<"Void", "void"> {
  let summary = "XB void type";
  let assemblyFormat = "";
  let genVerifyDecl = 0;
}

#endif
