#ifndef XBLANG_DIALECT_XBLANG_IR_DECL_TD
#define XBLANG_DIALECT_XBLANG_IR_DECL_TD

include "xblang/Dialect/XBLang/IR/Dialect.td"
include "xblang/Dialect/XBLang/IR/Enums.td"
include "xblang/Dialect/XBLang/IR/Type.td"
include "xblang/Dialect/XBLang/IR/Interfaces.td"

include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"

//===----------------------------------------------------------------------===//
// XB func Op
//===----------------------------------------------------------------------===//

def FunctionOp: XBLang_Op<"func", [
  AutomaticAllocationScope, CallableOpInterface, FunctionOpInterface,
  IsolatedFromAbove, Symbol]> {
  let summary = "";
  let description = [{ }];
  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<StrAttr>:$sym_visibility,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs
                       );
  let regions = (region AnyRegion:$body);
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
    CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)
  >];
  let extraClassDeclaration = [{
    ::mlir::Region *getCallableRegion() {
      return isExternal() ? nullptr : &getBody();
    }
    ArrayRef<Type> getCallableResults() {
      return getFunctionType().getResults();
    }
    ::mlir::ArrayAttr getCallableArgAttrs() {
      return getArgAttrs().value_or(nullptr);
    }
    ::mlir::ArrayAttr getCallableResAttrs() {
      return getResAttrs().value_or(nullptr);
    }
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
    LogicalResult verifyType();
    bool isDeclaration() { return isExternal(); }
  }];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// XB var Op
//===----------------------------------------------------------------------===//

def VarOp: XBLang_Op<"var", [ImplicitCastTrait, 
                            DeclareOpInterfaceMethods<OpAsmOpInterface,
                                                      ["getAsmResultNames"]>]> {
  let summary = "Return from function";
  let description = [{}];
  let arguments = (ins 
    SymbolNameAttr:$sym_name,
    TypeAttr:$type,
    VarKindEnum:$kind,
    Optional<AnyType>:$init
  );
  let results = (outs AnyType:$decl);
  let assemblyFormat = [{
    `[` $kind `]` $sym_name `:` $type `->` type($decl)
      (` ``[` ` ` `=` $init^ `:` type($init)`]`)? attr-dict
  }];
  let hasVerifier = 0;
}

#endif
