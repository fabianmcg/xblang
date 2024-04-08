//===- Concepts.td - Base XBLang language concepts  -----------*- Tblgen-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines base XBLang language concepts
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_XLG_CONCEPTSBASE_TD
#define XBLANG_XLG_CONCEPTSBASE_TD

include "xblang/XLG/XLGBase.td"
include "xblang/Interfaces/Symbol.td"
include "xblang/XLG/Interfaces.td"
include "mlir/IR/SymbolInterfaces.td"


//===----------------------------------------------------------------------===//
// Symbol concepts
//===----------------------------------------------------------------------===//

def SymbolCep : Concept<"Symbol", "sym"> {
  let summary = "Symbol concept";
  let traits = [SymbolInterface];
  let args = (ins SymbolNameAttr:$sym_id, OptionalAttr<SymbolNameAttr>:$usr);
}

def SymbolTableCep : Concept<"SymbolTable", "sym_table"> {
  let summary = "Symbol table concept";
  let traits = [SymbolTableInterface];
}

def SymbolRefCep : Concept<"SymbolRef", "sym_ref"> {
  let summary = "Symbol reference concept";
}

//===----------------------------------------------------------------------===//
// Attr concepts
//===----------------------------------------------------------------------===//

def AttrCep : Concept<"Attr", "attr"> {
  let summary = "xlg.attr";
}

//===----------------------------------------------------------------------===//
// Type concepts
//===----------------------------------------------------------------------===//

def TypeCep : Concept<"Type", "type"> {
  let summary = "xlg.type";
  let args = (ins OptionalAttr<TypeAttr>:$type);
  let traits = [TypeAttrInterface];
}

def BuiltinTypeCep : Concept<"BuiltinType", "builtin_type", [TypeCep]> {
}

def DerivedTypeCep : Concept<"DerivedType", "derived_type", [TypeCep]> {
}

//===----------------------------------------------------------------------===//
// Type concepts
//===----------------------------------------------------------------------===//

def TypedCep : Concept<"TypedConcept", "typed_node"> {
  let args = (ins OptionalAttr<TypeAttr>:$type);
  let traits = [TypeAttrInterface];
}

//===----------------------------------------------------------------------===//
// Decl concepts
//===----------------------------------------------------------------------===//

def DeclCep : Concept<"Decl", "decl">;

def NamedDeclCep : Concept<"NamedDecl", "named_decl", [DeclCep, SymbolCep]>;

def ModuleDeclCep :
    Concept<"ModuleDecl", "module_decl", [NamedDeclCep, SymbolTableCep]> {
}

def ValueDeclCep : Concept<"ValueDecl", "value_decl", [NamedDeclCep, TypedCep]>;

def TypeDeclCep : Concept<"TypeDecl", "type_decl", [NamedDeclCep, TypeCep]>;

def VarDeclCep : Concept<"VarDecl", "var_decl", [ValueDeclCep]> {
  let args = (ins Optional<TypeCep>:$valueType,
                  Optional<XLGConceptClass<"::xblang::xlg::Expr">>:$expr);
  let traits = [AttrSizedOperandSegments];
}

def FuncDeclCep :
    Concept<"FuncDecl", "func_decl", [ValueDeclCep, SymbolTableCep]> {
}

def FuncDefCep :
    Concept<"FuncDef", "func_def", [DeclCep, TypeCep]> {
  let args = (ins Optional<TypeCep>:$returnType,
                  Variadic<VarDeclCep>:$arguments);
  let traits = [AttrSizedOperandSegments];
}

def ObjectDeclCep :
    Concept<"ObjectDecl", "object_decl", [TypeDeclCep, SymbolTableCep]> {
}

//===----------------------------------------------------------------------===//
// Statement concepts
//===----------------------------------------------------------------------===//

def StmtCep : Concept<"Stmt", "stmt">;

def ControlFlowStmtCep : Concept<"ControlFlowStmt", "control_flow", [StmtCep]>;

def LoopStmtCep : Concept<"LoopStmt", "loop", [ControlFlowStmtCep]>;

def CFTerminatorStmtCep :
    Concept<"CFTerminatorStmt", "cf_terminator", [ControlFlowStmtCep]>;

def ReturnStmtCep : Concept<"ReturnStmt", "return", [CFTerminatorStmtCep]>;

def ExprCep : Concept<"Expr", "expr", [StmtCep, TypedCep]>;

def ConstExprCep : Concept<"ConstExpr", "const_expr", [ExprCep]> {
  let args = (ins OptionalAttr<AnyAttr>:$expr);
}

def CallExprCep : Concept<"CallExpr", "call_expr", [ExprCep]> {
  let args = (ins ExprCep:$callee, Variadic<ExprCep>:$arguments);
}

def CastExprCep : Concept<"CastExpr", "cast_expr", [ExprCep]> {
  let args = (ins ExprCep:$expr, Optional<TypeCep>:$dstType);
}

def RefExprCep : Concept<"RefExpr", "ref_expr", [ExprCep]> {
}

def SymbolRefExprCep : Concept<"SymbolRefExpr", "symbol_ref_expr", [RefExprCep]> {
  let args = (ins SymbolRefAttr:$sym_name, UnitAttr:$delayedResolution);
}

def ValueRefExprCep : Concept<"ValueRefExpr", "value_ref_expr", [RefExprCep]> {
  let args = (ins ValueDeclCep:$value);
}

#endif // XBLANG_XLG_CONCEPTSBASE_TD
