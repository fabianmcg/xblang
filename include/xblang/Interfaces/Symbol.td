//===- Interfaces.td - Basic MLIR interfaces  -----------------*- Tblgen-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines base MLIR interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_BASIC_INTERFACES_TD
#define XBLANG_BASIC_INTERFACES_TD

include "mlir/IR/Interfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Symbol interfaces
//===----------------------------------------------------------------------===//

def SymbolVisibilityEnum : I32EnumAttr<"SymbolVisibility", "Symbol visibility kind", [
    I32EnumAttrCase<"Public", 0, "public">,
    I32EnumAttrCase<"Protected", 1, "protected">,
    I32EnumAttrCase<"Private", 2, "private">,
    I32EnumAttrCase<"Nested", 3, "nested">,
  ]> {
  let cppNamespace = "::xblang";
}

def SymbolPropertiesEnum : I32EnumAttr<"SymbolProperties",
                                        "Symbol properties", [
    I32EnumAttrCase<"Unique", 0, "unique">,
    I32EnumAttrCase<"Overridable", 1, "overridable">,
    I32EnumAttrCase<"TemplateInstance", 2, "template_instance">,
    I32EnumAttrCase<"Template", 3, "template">,
    I32EnumAttrCase<"Mergeable", 4, "mergeable">,
  ]> {
  let cppNamespace = "::xblang";
}

def SymbolInterface : OpInterface<"Symbol"> {
  let description = [{
    This interface describes an operation that may define a `Symbol`. A `Symbol`
    operation must define an string identifier, a symbol class and a symbol type.
    Additionally symbols can provide visibility information.
  }];
  let cppNamespace = "::xblang";
  let methods = [
    InterfaceMethod<"Returns the identifier attribute of this symbol.",
      "::mlir::StringAttr", "getIdentifier", (ins), [{}], [{
        return $_op.getSymIdAttr();
      }]
    >,
    InterfaceMethod<"Returns the unified symbol resolution identifier.",
      "::mlir::StringAttr", "getUSR", (ins), [{}], [{
        return $_op.getUsrAttr();
      }]
    >,
    InterfaceMethod<"Returns the visibility of this symbol.",
      "::xblang::SymbolVisibility", "getSymbolVisibility", (ins), [{}], [{
        return ::xblang::SymbolVisibility::Public;
      }]
    >,
    InterfaceMethod<"Returns the symbol properties.",
      "::xblang::SymbolProperties", "getSymbolProps", (ins), [{}], [{
        return SymbolProperties::Unique;
      }]
    >
  ];
  let verify = [{
    return ::xblang::verifySymbol($_op);
  }];
}

def SymbolTableKindEnum : I32EnumAttr<"SymbolTableKind", "Symbol table kind", [
    I32EnumAttrCase<"Unordered", 0, "unordered">,
    I32EnumAttrCase<"Ordered", 1, "ordered">
  ]> {
  let cppNamespace = "::xblang";
}

def SymbolTablePropertiesEnum : I32EnumAttr<"SymbolTableProperties",
                                            "Symbol table properties", [
    I32EnumAttrCase<"None", 0, "none">,
    I32EnumAttrCase<"Abstract", 1, "abstract">,
    I32EnumAttrCase<"Mergeable", 2, "mergeable">
  ]> {
  let cppNamespace = "::xblang";
}

def SymbolTableInterface : OpInterface<"SymbolTableInterface"> {
  let description =  [{
    This interface describes an operation that may define a `SymbolTable`. A
    `SymbolTable` operation can store a collection of `Symbol`s and provide
    methods for looking up symbols.
  }];
  let cppNamespace = "::xblang";
  let methods = [
    InterfaceMethod<"Returns the kind of table used to store the symbols.",
      "::xblang::SymbolTableKind", "getSymbolTableKind", (ins), [{}], [{
        return ::xblang::SymbolTableKind::Unordered;
      }]
    >,
    InterfaceMethod<"Returns the symbol table properties.",
      "::xblang::SymbolTableProperties", "getSymbolTableProps", (ins), [{}], [{
        return ::xblang::SymbolTableProperties::None;
      }]
    >
  ];
  let verify = [{
    return ::xblang::verifySymbolTable($_op);
  }];
}

#endif // XBLANG_BASIC_INTERFACES_TD
