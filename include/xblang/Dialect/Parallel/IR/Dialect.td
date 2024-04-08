#ifndef XBLANG_DIALECT_PARALLEL_IR_DIALECT_TD
#define XBLANG_DIALECT_PARALLEL_IR_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Parallel IR dialect definition.
//===----------------------------------------------------------------------===//

def Par_Dialect : Dialect {
  let name = "par";
  let summary = "Parallel IR MLIR dialect.";
  let description = [{}];
  let cppNamespace = "::mlir::par";
  let useDefaultAttributePrinterParser = 1;
  let dependentDialects = [
    "::mlir::gpu::GPUDialect"
  ];
  let usePropertiesForAttributes = 1;
}

//===----------------------------------------------------------------------===//
// Base par classes definition.
//===----------------------------------------------------------------------===//

class Par_Op<string mnemonic, list<Trait> traits = []>
    : Op<Par_Dialect, mnemonic, traits>;

class Par_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Par_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

class Par_Attr<string name, string attrMnemonic, list<Trait> traits = []>:
  AttrDef<Par_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

#endif
