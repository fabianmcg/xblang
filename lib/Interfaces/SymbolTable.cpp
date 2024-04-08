//===- SymbolTable.cpp - Symbol table ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares symbol table related classes.
//
//===----------------------------------------------------------------------===//

#include "xblang/Interfaces/SymbolTable.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "xblang/Support/Format.h"
#include "xblang/Support/Worklist.h"
#include "llvm/Support/Debug.h"
#include <stack>

using namespace xblang;

#define DEBUG_TYPE "symbol-table"

mlir::LogicalResult xblang::verifySymbol(mlir::Operation *op) {
  Symbol symbol = llvm::dyn_cast_or_null<Symbol>(op);
  if (!symbol)
    return mlir::failure();
  SymbolProperties props = symbol.getSymbolProps();
  if (props == SymbolProperties::Template ||
      props == SymbolProperties::Mergeable) {
    SymbolTableInterface symTable =
        llvm::dyn_cast_or_null<SymbolTableInterface>(op);
    if (!symTable)
      return op->emitError("symbol should also be a symbol table");
  }
  return mlir::success();
}

mlir::LogicalResult xblang::verifySymbolTable(mlir::Operation *op) {
  SymbolTableInterface symTable =
      llvm::dyn_cast_or_null<SymbolTableInterface>(op);
  if (!symTable)
    return mlir::failure();
  if (symTable.getSymbolTableProps() == SymbolTableProperties::Abstract) {
    SymbolTableInterface parentTable =
        llvm::dyn_cast_or_null<SymbolTableInterface>(op->getParentOp());
    if (!parentTable)
      return op->emitError("parent table is not a symbol table and this table "
                           "is marked as abstract");
  }
  if (symTable.getSymbolTableProps() != SymbolTableProperties::None) {
    Symbol symbol = llvm::dyn_cast_or_null<Symbol>(op);
    if (!symbol)
      return op->emitError("symbol table should also be a symbol");
  }
  return mlir::success();
}

namespace {
//===----------------------------------------------------------------------===//
// Symbol group
//===----------------------------------------------------------------------===//
/// Holds a group of symbols with the same identifier.
class SymbolGroup : public llvm::SmallVector<SymbolInstance> {
public:
  SymbolGroup(SymbolInstance instance) {
    if (instance)
      push_back(instance);
  }
};

//===----------------------------------------------------------------------===//
// Ordered symbol table
//===----------------------------------------------------------------------===//
class OrderedSymbolTable : public SymbolTable {
private:
  DenseMap<mlir::Attribute, std::unique_ptr<SymbolInstance>> symbols;

public:
  OrderedSymbolTable(SymbolTableContext *context, mlir::Operation *op,
                     SymbolTable *parent)
      : SymbolTable(context, op, parent, SymbolTableKind::Ordered) {}

  /// Inserts a new symbol to the table.
  SymbolCollection insert(SymbolInstance);
  /// Erases a symbol collection.
  void erase(mlir::Attribute key);
  /// Erases a symbol instance.
  void erase(SymbolInstance);
  /// Looks up a symbol collection.
  SymbolCollection lookup(mlir::Attribute key) const;
  /// Clears all the symbols from the table.
  void clear();
};

//===----------------------------------------------------------------------===//
// Unordered symbol table
//===----------------------------------------------------------------------===//
class UnorderedSymbolTable : public SymbolTable {
private:
  DenseMap<mlir::Attribute, std::unique_ptr<SymbolGroup>> symbols;

public:
  UnorderedSymbolTable(SymbolTableContext *context, mlir::Operation *op,
                       SymbolTable *parent)
      : SymbolTable(context, op, parent, SymbolTableKind::Unordered) {}

  /// Inserts a new symbol to the table.
  SymbolCollection insert(SymbolInstance);
  /// Erases a symbol collection.
  void erase(mlir::Attribute key);
  /// Erases a symbol instance.
  void erase(SymbolInstance);
  /// Looks up a symbol collection.
  SymbolCollection lookup(mlir::Attribute key) const;
  /// Clears all the symbols from the table.
  void clear();
};
} // namespace

//===----------------------------------------------------------------------===//
// Symbol collection
//===----------------------------------------------------------------------===//

mlir::StringAttr SymbolCollection::getIdentifierAttr() const {
  if (size() > 0)
    return front().getIdentifierAttr();
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Symbol table
//===----------------------------------------------------------------------===//
SymbolTable::~SymbolTable() = default;

std::unique_ptr<SymbolTable> SymbolTable::get(SymbolTableContext *context,
                                              mlir::Operation *op,
                                              SymbolTable *parent,
                                              SymbolTableKind tableKind) {
  if (tableKind == SymbolTableKind::Ordered)
    return std::unique_ptr<SymbolTable>(
        new OrderedSymbolTable(context, op, parent));
  return std::unique_ptr<SymbolTable>(
      new UnorderedSymbolTable(context, op, parent));
}

SymbolCollection SymbolTable::insert(SymbolInstance symbolInstance) {
  LLVM_DEBUG({
    auto op = symbolInstance.getSymbol().getOperation();
    llvm::dbgs() << fmt(
        "Symbol table: insert symbol `{0}` '{1}'({2}) to table ({3})\n",
        symbolInstance.getIdentifier(), op->getName().getStringRef(), op, this);
  });
  if (tableKind == SymbolTableKind::Ordered)
    return static_cast<OrderedSymbolTable &>(*this).insert(symbolInstance);
  return static_cast<UnorderedSymbolTable &>(*this).insert(symbolInstance);
}

void SymbolTable::erase(mlir::StringAttr key) {
  if (tableKind == SymbolTableKind::Ordered)
    return static_cast<OrderedSymbolTable &>(*this).erase(key);
  return static_cast<UnorderedSymbolTable &>(*this).erase(key);
}

void SymbolTable::erase(SymbolInstance symbolInstance) {
  if (tableKind == SymbolTableKind::Ordered)
    return static_cast<OrderedSymbolTable &>(*this).erase(symbolInstance);
  return static_cast<UnorderedSymbolTable &>(*this).erase(symbolInstance);
}

namespace {
SymbolCollection lookup(const SymbolTable *table, mlir::StringAttr key) {
  if (table->getKind() == SymbolTableKind::Ordered)
    return static_cast<const OrderedSymbolTable &>(*table).lookup(key);
  return static_cast<const UnorderedSymbolTable &>(*table).lookup(key);
}

mlir::StringAttr getAttr(mlir::StringAttr attr) { return attr; }

mlir::StringAttr getAttr(mlir::FlatSymbolRefAttr attr) {
  return attr.getAttr();
}

template <typename T>
SymbolCollection lookup(const SymbolTable *table,
                        mlir::ArrayRef<T> identifier) {
  size_t index = 0;
  SymbolCollection collection;
  while (index < identifier.size()) {
    // If the table is null, then the lookup failed.
    if (!table)
      return {};
    collection = table->lookup(getAttr(identifier[index++]));
    // Break if we have consumed the identifier or failed to find the next
    // symbol.
    if (index == identifier.size() || collection.empty())
      break;
    // Qualified lookups are only supported on collections with only one symbol
    // or in mergeable tables.
    if (collection.size() != 1 && collection[0].getSymbol().getSymbolProps() !=
                                      SymbolProperties::Mergeable)
      return {};
    // Get the next table.
    table = table->findTable(collection[0].getSymbol().getOperation());
  }
  return collection;
}
} // namespace

SymbolCollection SymbolTable::lookup(mlir::StringAttr key, bool local) const {
  if (local)
    return ::lookup(this, key);
  SymbolTable const *table = this;
  while (table) {
    SymbolCollection collection = ::lookup(table, key);
    if (collection.size() > 0)
      return collection;
    table = table->getParent();
  }
  return {};
}

const SymbolTable *SymbolTable::lookupRoot(mlir::StringAttr root) const {
  const SymbolTable *table = this;
  if (root == nullptr) {
    // Find the root table.
    while (table) {
      if (table->getParent() == nullptr)
        break;
      table = table->getParent();
    }
  } else {
    SymbolCollection collection = lookup(root, false);
    // Return if the lookup failed.
    if (collection.empty())
      return {};
    // Symbol lookups with more than one symbol are only supported in mergeable
    // tables.
    if (collection.size() != 1 && collection[0].getSymbol().getSymbolProps() !=
                                      SymbolProperties::Mergeable) {
      return {};
    }
    table = findTable(collection[0].getSymbol().getOperation());
  }
  return table;
}

SymbolCollection
SymbolTable::lookup(mlir::StringAttr root,
                    mlir::ArrayRef<mlir::StringAttr> nestedReferences) const {
  if (nestedReferences.empty())
    return root ? lookup(root, false) : SymbolCollection{};
  return ::lookup(lookupRoot(root), nestedReferences);
}

SymbolCollection SymbolTable::lookup(
    mlir::StringAttr root,
    mlir::ArrayRef<mlir::FlatSymbolRefAttr> nestedReferences) const {
  if (nestedReferences.empty())
    return root ? lookup(root, false) : SymbolCollection{};
  return ::lookup(lookupRoot(root), nestedReferences);
}

void SymbolTable::clear() {
  if (tableKind == SymbolTableKind::Ordered)
    return static_cast<OrderedSymbolTable &>(*this).clear();
  return static_cast<UnorderedSymbolTable &>(*this).clear();
}

SymbolTable *SymbolTable::findTable(mlir::Operation *op) const {
  assert(context && "null symbol table context");
  return context->get(op);
}

mlir::Operation *SymbolTable::lookupUSR(mlir::Attribute usr) const {
  return context ? context->lookupUSR(usr) : nullptr;
}

//===----------------------------------------------------------------------===//
// Symbol table context
//===----------------------------------------------------------------------===//

mlir::LogicalResult SymbolTableContext::buildTables(mlir::Operation *op,
                                                    SymbolTable *parent) {
  if (!op)
    return mlir::failure();
  IRWorklist list({op});
  std::stack<std::pair<SymbolTable *, mlir::Operation *>> stack;
  if (parent)
    stack.push({parent, nullptr});
  while (list.size() > 0) {
    IRWorklistElement top = list.pop();
    if (top.getCount() > 2) {
      if (!stack.empty() && stack.top().second == top.get())
        stack.pop();
      continue;
    }
    SymbolTable *table =
        insert(top.get(), stack.empty() ? nullptr : stack.top().first);
    if (stack.size() > 0) {
      if (auto sym = dyn_cast<Symbol>(top.get())) {
        if (table == stack.top().first) {
          if (table->getParent()->insert(sym).empty())
            return mlir::failure();
          table = nullptr;
        } else if (stack.top().first->insert(sym).empty())
          return mlir::failure();
      }
    }
    if (top.get()->getNumRegions() == 0)
      continue;
    list.push_back(++top);
    list.addOp(top.get());
    if (table)
      stack.push({table, top.get()});
  }
  return success();
}

mlir::FailureOr<SymbolTableContext>
SymbolTableContext::create(mlir::Operation *op) {
  if (!op)
    return mlir::failure();
  SymbolTableContext tables;
  if (failed(tables.buildTables(op)))
    return failure();
  return std::move(tables);
}

SymbolTable *SymbolTableContext::insert(mlir::Operation *op,
                                        SymbolTable *parent) {
  if (!op)
    return nullptr;
  SymbolTable *table = nullptr;
  if (op->hasTrait<mlir::OpTrait::SymbolTable>())
    table = getOrCreateTable(op, parent, SymbolTableKind::Unordered,
                             SymbolTableProperties::None)
                .get();
  else if (auto iface = mlir::dyn_cast<SymbolTableInterface>(op))
    table = getOrCreateTable(op, parent, iface.getSymbolTableKind(),
                             iface.getSymbolTableProps())
                .get();
  if (table) {
    LLVM_DEBUG({
      llvm::dbgs() << fmt(
          "Symbol table: create table ({0}) for op '{1}'({2})\n", table,
          op->getName().getStringRef(), op);
    });
  }
  return table;
}

std::shared_ptr<SymbolTable> SymbolTableContext::find(mlir::Operation *op) {
  if (!op)
    return nullptr;
  auto it = tableContext.find(op);
  if (it == tableContext.end())
    return nullptr;
  return it->second;
}

std::shared_ptr<SymbolTable>
SymbolTableContext::getOrCreateTable(mlir::Operation *op, SymbolTable *parent,
                                     SymbolTableKind kind,
                                     SymbolTableProperties props) {
  using ptr_t = std::shared_ptr<SymbolTable>;
  ptr_t &table = tableContext[op];
  if (table)
    return table;
  // Create a new table if it's a regular symbol table.
  if (props == SymbolTableProperties::None)
    return table = ptr_t(SymbolTable::get(this, op, parent, kind));
  if (!parent) {
    op->emitError("null parent symbol table");
    return nullptr;
  }
  // Abstract tables use the parent table for storing symbols.
  if (props == SymbolTableProperties::Abstract) {
    table = find(parent);
    if (!table) {
      op->emitError("parent of abstract table is not present");
      return nullptr;
    }
    return table;
  }
  // The symbol must be mergeable.
  Symbol symbol = cast<Symbol>(op);
  SymbolCollection collection = parent->lookup(symbol.getIdentifier(), true);
  // If the symbol couldn't be found add a new table, otherwise look for the
  // symbol symbol table.
  if (collection.empty())
    return table = ptr_t(SymbolTable::get(this, op, parent, kind));
  else {
    table = find(collection[0].getSymbol().getOperation());
    if (!table) {
      op->emitError("shared table is not present");
      return nullptr;
    }
  }
  return table;
}

//===----------------------------------------------------------------------===//
// Ordered symbol table
//===----------------------------------------------------------------------===//

inline SymbolCollection
OrderedSymbolTable::insert(SymbolInstance symbolInstance) {
  if (!symbolInstance)
    return {};
  std::unique_ptr<SymbolInstance> &sym =
      symbols[symbolInstance.getIdentifierAttr()];
  if (!sym)
    sym.reset(new SymbolInstance(symbolInstance));
  else
    *sym = symbolInstance;
  return SymbolCollection(sym.get(), 1);
}

inline void OrderedSymbolTable::erase(mlir::Attribute key) {
  symbols.erase(key);
}

inline void OrderedSymbolTable::erase(SymbolInstance symbolInstance) {
  erase(symbolInstance.getIdentifierAttr());
}

inline SymbolCollection OrderedSymbolTable::lookup(mlir::Attribute key) const {
  auto it = symbols.find(key);
  if (it == symbols.end())
    return {};
  return SymbolCollection(it->second.get(), 1);
}

inline void OrderedSymbolTable::clear() { symbols.clear(); }

//===----------------------------------------------------------------------===//
// Unordered symbol table
//===----------------------------------------------------------------------===//

inline SymbolCollection
UnorderedSymbolTable::insert(SymbolInstance symbolInstance) {
  if (!symbolInstance)
    return {};
  std::unique_ptr<SymbolGroup> &sym =
      symbols[symbolInstance.getIdentifierAttr()];
  if (!sym) {
    sym.reset(new SymbolGroup(symbolInstance));
    return *sym;
  }
  SymbolProperties props = symbolInstance.getProperties();
  if (props == SymbolProperties::Unique) {
    symbolInstance->emitError("invalid redefinition of symbol");
    sym->front()->emitError("first defined here");
    return {};
  } else if (props == SymbolProperties::Mergeable) {
    sym->push_back(symbolInstance);
  } else if (props == SymbolProperties::Template) {
    sym->push_back(symbolInstance);
  } else {
    size_t i = sym->size();
    while (i >= 1) {
      if ((*sym)[i - 1].getProperties() == props)
        break;
      i--;
    }
    sym->insert(sym->begin() + i, symbolInstance);
  }
  return *sym;
}

inline void UnorderedSymbolTable::erase(mlir::Attribute key) {
  symbols.erase(key);
}

inline void UnorderedSymbolTable::erase(SymbolInstance symbolInstance) {
  if (!symbolInstance)
    return;
  auto it = symbols.find(symbolInstance.getIdentifierAttr());
  if (it == symbols.end())
    return;
  auto git = std::find(it->second->begin(), it->second->end(), symbolInstance);
  it->second->erase(git);
}

inline SymbolCollection
UnorderedSymbolTable::lookup(mlir::Attribute key) const {
  auto it = symbols.find(key);
  if (it == symbols.end())
    return {};
  return *(it->second);
}

inline void UnorderedSymbolTable::clear() { symbols.clear(); }

#include "xblang/Interfaces/SymbolEnums.cpp.inc"

#include "xblang/Interfaces/SymbolInterfaces.cpp.inc"
