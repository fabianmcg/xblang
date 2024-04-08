//===- SymbolTable.h - Symbol table -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines symbol table related classes and interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_INTERFACES_SYMBOLTABLE_H
#define XBLANG_INTERFACES_SYMBOLTABLE_H

#include "mlir/IR/OpDefinition.h"
#include "xblang/Basic/Concept.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

namespace xblang {
mlir::LogicalResult verifySymbol(mlir::Operation *op);
mlir::LogicalResult verifySymbolTable(mlir::Operation *op);
} // namespace xblang

#include "xblang/Interfaces/SymbolEnums.h.inc"

#include "xblang/Interfaces/SymbolInterfaces.h.inc"

namespace xblang {
class SymbolTable;
class SymbolTableContext;

//===----------------------------------------------------------------------===//
// Symbol instance
//===----------------------------------------------------------------------===//
/// Class for holding instances of symbols.
class SymbolInstance {
public:
  SymbolInstance(Symbol symbol) : symbol(symbol) {}

  operator bool() const { return symbol != nullptr; }

  bool operator==(const SymbolInstance &other) const {
    return symbol == other.symbol;
  }

  /// Returns the identifier of this symbol.
  llvm::StringRef getIdentifier() const {
    if (auto attr = getIdentifierAttr())
      return attr.getValue();
    return "";
  }

  /// Returns the identifier attribute of this symbol.
  mlir::StringAttr getIdentifierAttr() const {
    return symbol ? symbol.getIdentifier() : nullptr;
  }

  /// Returns the symbol visibility.
  SymbolVisibility getVisibility() const {
    return symbol.getSymbolVisibility();
  }

  /// Returns the symbol properties.
  SymbolProperties getProperties() const { return symbol.getSymbolProps(); }

  /// Returns the underlying operation.
  mlir::Operation *getOp() const { return symbol.getOperation(); }

  mlir::Operation *operator->() const { return symbol.getOperation(); }

  /// Returns the symbol being held.
  Symbol getSymbol() const { return symbol; }

private:
  SymbolInstance() = default;
  mutable Symbol symbol{};
};

//===----------------------------------------------------------------------===//
// Symbol collection
//===----------------------------------------------------------------------===//
/// A reference to a collection of symbols.
class SymbolCollection : public llvm::ArrayRef<SymbolInstance> {
public:
  using Base = llvm::ArrayRef<SymbolInstance>;
  using Base::Base;

  /// Returns the identifier attribute of this symbol.
  mlir::StringAttr getIdentifierAttr() const;

  /// Returns the identifier of this symbol.
  llvm::StringRef getIdentifier() const {
    if (auto attr = getIdentifierAttr())
      return attr.getValue();
    return "";
  }
};

//===----------------------------------------------------------------------===//
// Symbol table
//===----------------------------------------------------------------------===//
/// Base class for all symbol tables. Symbol tables can only be obtained via
/// symbol table context.
class SymbolTable {
private:
  friend class SymbolTableContext;

  SymbolTableContext *context;
  mlir::Operation *op;
  SymbolTable *parent = nullptr;
  SymbolTableKind tableKind{};

  /// Creates a new symbol symbol table.
  static std::unique_ptr<SymbolTable> get(SymbolTableContext *context,
                                          mlir::Operation *op,
                                          SymbolTable *parent,
                                          SymbolTableKind kind);

protected:
  SymbolTable(SymbolTableContext *context, mlir::Operation *op,
              SymbolTable *parent, SymbolTableKind tableKind)
      : context(context), op(op), parent(parent), tableKind(tableKind) {}

  SymbolTable(const SymbolTable &) = delete;

public:
  virtual ~SymbolTable() = 0;
  /// Inserts a new symbol to the table.
  SymbolCollection insert(SymbolInstance);
  /// Erases a symbol instance.
  void erase(SymbolInstance);
  /// Erases a symbol collection.
  void erase(mlir::StringAttr key);

  void erase(llvm::StringRef key) {
    assert(op && "null op");
    erase(mlir::StringAttr::get(op->getContext(), key));
  }

  /// Looks up a symbol collection.
  SymbolCollection lookup(mlir::StringAttr key, bool local = false) const;

  SymbolCollection lookup(mlir::FlatSymbolRefAttr key,
                          bool local = false) const {
    return lookup(key.getAttr(), local);
  }

  SymbolCollection lookup(llvm::StringRef key, bool local = false) const {
    assert(op && "null op");
    return lookup(mlir::StringAttr::get(op->getContext(), key), local);
  }

  SymbolCollection
  lookup(mlir::StringAttr root,
         mlir::ArrayRef<mlir::StringAttr> nestedReferences) const;
  SymbolCollection
  lookup(mlir::StringAttr root,
         mlir::ArrayRef<mlir::FlatSymbolRefAttr> nestedReferences) const;

  SymbolCollection lookup(mlir::SymbolRefAttr key) const {
    return lookup(key.getRootReference(), key.getNestedReferences());
  }

  /// Clears all the symbols from the table.
  void clear();

  /// Returns the op owning this table.
  mlir::Operation *getOp() const { return op; }

  /// Returns the parent table.
  SymbolTable *getParent() const { return parent; }

  /// Sets the parent table.
  void setParent(SymbolTable *table) { parent = table; }

  /// Returns a symbol table for a given operation by looking through the
  /// context.
  SymbolTable *findTable(mlir::Operation *op) const;

  /// Returns the symbol table kind.
  SymbolTableKind getKind() const { return tableKind; }

  /// Returns the symbol table context if present.
  SymbolTableContext *getContext() const { return context; }

  /// Looks up an operation by the USR.
  mlir::Operation *lookupUSR(mlir::Attribute usr) const;

protected:
  /// Looks up the root table for a qualified search.
  const SymbolTable *lookupRoot(mlir::StringAttr root) const;
};

//===----------------------------------------------------------------------===//
// Symbol table context
//===----------------------------------------------------------------------===//
/// Class for storing a collection of symbol tables.
class SymbolTableContext {
public:
  SymbolTableContext() = default;

  /// Creates a symbol table context by traversing op.
  static mlir::FailureOr<SymbolTableContext> create(mlir::Operation *op);

  /// Recursively builds the symbol tables for an op with a given symbol table
  /// parent.
  mlir::LogicalResult buildTables(mlir::Operation *op,
                                  SymbolTable *parent = nullptr);

  /// Clears the stack.
  void clear() { tableContext.clear(); }

  /// Erases a symbol table from the context.
  void erase(mlir::Operation *op) { tableContext.erase(op); }

  /// Inserts a new symbol table.
  SymbolTable *insert(mlir::Operation *op, SymbolTable *parent);

  /// Returns the symbol table mapped to op or nullptr.
  SymbolTable *get(mlir::Operation *op) const {
    auto it = tableContext.find(op);
    if (it == tableContext.end())
      return nullptr;
    return it->second.get();
  }

  /// Looks up an operation by the USR.
  mlir::Operation *lookupUSR(mlir::Attribute usr) const {
    return usrMap.lookup(usr);
  }

  /// Inserts an operation into the USR map.
  bool insertUSR(mlir::Attribute usr, mlir::Operation *op) {
    return usrMap.insert({usr, op}).second;
  }

  /// Sets the USR map to an specific operation.
  void setUSR(mlir::Attribute usr, mlir::Operation *op) { usrMap[usr] = op; }

private:
  /// Gets or creates a new symbol table.
  std::shared_ptr<SymbolTable> getOrCreateTable(mlir::Operation *op,
                                                SymbolTable *parent,
                                                SymbolTableKind kind,
                                                SymbolTableProperties props);

  /// Returns the symbol table container mapped to op or nullptr.
  std::shared_ptr<SymbolTable> find(mlir::Operation *op);

  std::shared_ptr<SymbolTable> find(SymbolTable *table) {
    if (!table)
      return {};
    return find(table->getOp());
  }

  llvm::DenseMap<mlir::Operation *, std::shared_ptr<SymbolTable>> tableContext;
  llvm::DenseMap<mlir::Attribute, mlir::Operation *> usrMap;
};
} // namespace xblang

#endif // XBLANG_INTERFACES_SYMBOLTABLE_H
