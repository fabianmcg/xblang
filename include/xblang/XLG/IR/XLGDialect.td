#ifndef XBLANG_XLG_DIALECT
#define XBLANG_XLG_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// XLG IR dialect definition.
//===----------------------------------------------------------------------===//

def XLG_Dialect : Dialect {
  let name = "xlg";
  let summary = "XLG MLIR dialect.";
  let cppNamespace = "::xblang::xlg";
  let dependentDialects = ["::xblang::XBContextDialect"];
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 0;
}

//===----------------------------------------------------------------------===//
// Base xlg classes definition.
//===----------------------------------------------------------------------===//

class XLG_Op<string mnemonic, list<Trait> traits = []>
    : Op<XLG_Dialect, mnemonic, traits>;

class XLG_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<XLG_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

class XLG_Attr<string name, string attrMnemonic, list<Trait> traits = []>:
  AttrDef<XLG_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// XLG types definition.
//===----------------------------------------------------------------------===//

def XLG_ConceptType : XLG_Type<"Concept", "concept"> {
  let summary = "Type for XLG expressions";
  let description = [{
    XLG:

    ```mlir
    concept ::= concept<`kind`>
    ```
  }];
  let parameters = (ins TypeParameter<"::xblang::ConceptContainer", "">:$conceptClass);
  let assemblyFormat = "`<` custom<ConceptType>($conceptClass) `>`";
}

//===----------------------------------------------------------------------===//
// XLG attrs definition.
//===----------------------------------------------------------------------===//

#endif // XBLANG_XLG_DIALECT
