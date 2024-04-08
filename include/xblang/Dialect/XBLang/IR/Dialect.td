#ifndef XBLANG_DIALECT_XBLANG_IR_DIALECT_TD
#define XBLANG_DIALECT_XBLANG_IR_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// XBLang IR dialect definition.
//===----------------------------------------------------------------------===//

def XBLang_Dialect : Dialect {
  let name = "xb";
  let summary = "XBLang IR MLIR dialect.";
  let description = [{}];
  let cppNamespace = "::xblang::xb";
  let useDefaultTypePrinterParser = 0;
//  let useDefaultAttributePrinterParser = 1;
  let hasConstantMaterializer = 1;
  let extraClassDeclaration = [{
    Type parseType(DialectAsmParser &p) const override;
    void printType(Type, DialectAsmPrinter &p) const override;
    void registerTypes();
  }];
}

//===----------------------------------------------------------------------===//
// Base xblang operation definition.
//===----------------------------------------------------------------------===//

class XBLang_Op<string mnemonic, list<Trait> traits = []> : Op<XBLang_Dialect, mnemonic, traits>;

class XBLang_Attr<string name, string attrMnemonic, list<Trait> traits = []>:
  AttrDef<XBLang_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

class XBLang_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<XBLang_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

class XBLangTypeTrait<string name>: NativeTrait<name # "Trait", ""> {
  let cppNamespace = "::xblang::xb";
}

def FundamentalType : XBLangTypeTrait<"FundamentalType">;
def ArithmeticType : XBLangTypeTrait<"ArithmeticType">;
def ScalarType : XBLangTypeTrait<"ScalarType">;
def AlgebraicType : XBLangTypeTrait<"AlgebraicType">;

#endif
