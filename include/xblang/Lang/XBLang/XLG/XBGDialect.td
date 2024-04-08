#ifndef XBLANG_XBG_DIALECT
#define XBLANG_XBG_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "xblang/XLG/XLGBase.td"

//===----------------------------------------------------------------------===//
// XBG IR dialect definition.
//===----------------------------------------------------------------------===//

def XBG_Dialect : Dialect {
  let name = "xbg";
  let summary = "XBG MLIR dialect.";
  let cppNamespace = "::xblang::xbg";
  let dependentDialects = [
    "::xblang::XBContextDialect",
    "::xblang::xlg::XLGDialect",
    "::xblang::xb::XBLangDialect"
  ];
  let useDefaultTypePrinterParser = 0;
  let useDefaultAttributePrinterParser = 0;
  let extraClassDeclaration = [{
    ::xblang::XBContext *getXBContext() const {
      assert(xblangContext && "invalid context");
      return xblangContext;
    }
    private:
    /// Initialize type constructs.
    void initializeType();
    /// Initialize decl constructs.
    void initializeDecl();
    /// Initialize stmt constructs.
    void initializeStmt();
    /// Initialize expr constructs.
    void initializeExpr();
    ::xblang::XBContext *xblangContext;
  }];
}

//===----------------------------------------------------------------------===//
// Base xbg classes definition.
//===----------------------------------------------------------------------===//

class XBG_Op<string mnemonic, list<Trait> traits = []>
    : Op<XBG_Dialect, mnemonic, traits>;

class XBG_Construct<string name, string mnemonic,
                    list<Concept> conceptList = [],
                    list<Trait> traits = []>
    : Construct<XBG_Dialect, name, mnemonic, conceptList, traits>;

class XBG_PureConstruct<string name, string mnemonic,
                        list<Concept> conceptList = []>
    : DialectConcept<XBG_Dialect, name, mnemonic, conceptList> {
  let pureConstruct = 1;
}

class XBG_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<XBG_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

class XBG_Attr<string name, string attrMnemonic, list<Trait> traits = []>:
  AttrDef<XBG_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

#endif // XBLANG_XBG_DIALECT
