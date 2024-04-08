#ifndef XBLANG_XLG_OPS
#define XBLANG_XLG_OPS

include "xblang/XLG/IR/XLGDialect.td"
include "xblang/Interfaces/Symbol.td"
include "xblang/XLG/Interfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// XLG Module Op
//===----------------------------------------------------------------------===//

def XLG_ModuleOp : XLG_Op<"module", [
    IsolatedFromAbove, NoRegionArguments, NoTerminator, OpAsmOpInterface,
    SingleBlock, Symbol
  ]> {
  let summary = "Creates an XLG module.";
  let description = [{
    Syntax:
    ```mlir
    xlg.module @myModule {
    }
    ```
  }];
  let arguments = (ins OptionalAttr<SymbolNameAttr>:$sym_name);
  let regions = (region SizedRegion<1>:$bodyRegion);
  let assemblyFormat = "attr-dict-with-keyword $bodyRegion";
  let extraClassDeclaration = [{
    /// Return the name of this module if present.
    std::optional<llvm::StringRef> getName() { return getSymName(); }

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    /// A ModuleOp may optionally define a symbol.
    bool isOptionalSymbol() { return true; }

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Allow the dialect prefix to be omitted.
    static llvm::StringRef getDefaultDialect() { return "xlg"; }
  }];
}

//===----------------------------------------------------------------------===//
// XLG Region Op
//===----------------------------------------------------------------------===//

def XLG_RegionOp : XLG_Op<"region", [
    NoRegionArguments, SingleBlock
  ]> {
  let summary = "Creates an XLG region.";
  let description = [{
    Syntax:
    ```mlir
    %region = xlg.region -> xlg.class<stmt> {
      ...
      return %0 : xlg.class<stmt>
    }
    ```
  }];
  let results = (outs XLG_ConceptType:$kind);
  let regions = (region SizedRegion<1>:$bodyRegion);
  let assemblyFormat = "attr-dict-with-keyword $bodyRegion `->` type($kind)";
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Allow the dialect prefix to be omitted.
    static llvm::StringRef getDefaultDialect() { return "xlg"; }

    //===------------------------------------------------------------------===//
    // XLG Methods
    //===------------------------------------------------------------------===//

    mlir::Operation* getBuilder();
    template <typename Op> Op getBuilderOp() {
      return mlir::dyn_cast_or_null<Op>(getBuilder());
    }
  }];
}

//===----------------------------------------------------------------------===//
// XLG RegionReturn Op
//===----------------------------------------------------------------------===//

def XLG_RegionReturnOp : XLG_Op<"ret", [
    Terminator, ReturnLike, Pure
  ]> {
  let summary = "Returns a value from a region.";
  let description = [{
    Syntax:
    ```mlir
    ret %0 : i32
    ```
  }];
  let arguments = (ins Variadic<AnyType>:$expr);
  let assemblyFormat = "($expr^ `:` type($expr))? attr-dict";
}

//===----------------------------------------------------------------------===//
// XLG Return Op
//===----------------------------------------------------------------------===//

def XLG_ReturnOp : XLG_Op<"return", [
    Terminator, ReturnLike, Pure
  ]> {
  let summary = "Returns the XLG node builder.";
  let description = [{
    Syntax:
    ```mlir
    return %0 : xlg.expr<#xbxlg.stmt>
    ```
  }];
  let arguments = (ins XLG_ConceptType:$expr);
  let assemblyFormat = "$expr `:` type($expr) attr-dict";
  let extraClassDeclaration = [{
    mlir::Operation* getBuilder();
    template <typename Op> Op getBuilderOp() {
      return mlir::dyn_cast<Op>(getBuilder());
    }
  }];
}

//===----------------------------------------------------------------------===//
// XLG SymbolRef Op
//===----------------------------------------------------------------------===//

def XLG_SymbolRefOp : XLG_Op<"symbol_ref"> {
  let summary = "XLG generic symbol reference";
  let description = [{
    Syntax:
    ```mlir
    %0 = xlg.symbol_ref @ref
    ```
  }];
  let arguments = (ins SymbolRefAttr:$sym_id);
  let results = (outs XLG_ConceptType:$res);
  let assemblyFormat = "$sym_id `:` type($res) attr-dict";
}

//===----------------------------------------------------------------------===//
// XLG Template Op
//===----------------------------------------------------------------------===//

def XLG_TemplateOp : XLG_Op<"template", [
    CallableOpInterface, IsolatedFromAbove, SingleBlock,
    DeclareOpInterfaceMethods<SymbolInterface, ["getSymbolProps"]>,
    DeclareOpInterfaceMethods<SymbolTableInterface, ["getSymbolTableKind",
                                                     "getSymbolTableProps"]>
  ]> {
  let summary = "Creates a XLG template.";
  let description = [{
    Syntax:
    ```mlir
    xlg.template @myVal -> xlg.expr<#xbg.stmt> {
    }
    ```
  }];
  let arguments = (ins SymbolNameAttr:$sym_id,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<SymbolNameAttr>:$usr);
  let regions = (region SizedRegion<1>:$bodyRegion);
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "::llvm::StringRef":$name,
                                 "::mlir::FunctionType":$type,
                                 CArg<"::mlir::Block*", "nullptr">:$body)>];
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    mlir::ArrayRef<mlir::Type> getArgumentTypes() {
      return getFunctionType().getInputs();
    }

    /// Returns the result types of this function.
    mlir::ArrayRef<mlir::Type> getResultTypes() {
      return getFunctionType().getResults();
    }

    /// Returns the region on the function operation that is callable.
    mlir::Region *getCallableRegion() { return &getBodyRegion(); }
  }];
}

//===----------------------------------------------------------------------===//
// XLG TemplateInstance Op
//===----------------------------------------------------------------------===//

def XLG_TemplateInstanceOp : XLG_Op<"template_instance"> {
  let summary = "Creates a XLG template instance.";
  let description = [{
    Syntax:
    ```mlir
    xlg.template_instance @sym (...) -> xlg.concept<func_decl>
    ```
  }];
  let arguments = (ins FlatSymbolRefAttr:$usr,
                       Variadic<XLG_ConceptType>:$arguments);
  let results = (outs XLG_ConceptType:$conceptClass);
  let regions = (region MaxSizedRegion<1>:$bodyRegion);
  let assemblyFormat = [{
    $usr `(` $arguments `:` type($arguments) `)`
    `->` type($conceptClass)
    attr-dict-with-keyword ($bodyRegion^)?
  }];
}

#endif // XBLANG_XLG_OPS
