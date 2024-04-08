//===- AnalizeSyntax.cpp - Process syntax pass -------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the syntax processing pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "xblang/Support/Format.h"
#include "xblang/Syntax/IR/SyntaxDialect.h"
#include "xblang/Syntax/Transforms/Passes.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/IntEqClasses.h"

#include <deque>

#define DEBUG_TYPE "analyze-syntax"

namespace xblang {
namespace syntaxgen {
#define GEN_PASS_DEF_ANALYZESYNTAX
#include "xblang/Syntax/Transforms/Passes.h.inc"
} // namespace syntaxgen
} // namespace xblang

using namespace mlir;
using namespace xblang;
using namespace xblang::syntaxgen;

namespace {
struct SymbolContext;
struct Terminal;
struct SymbolTuple;

/// Base class for all symbols.
struct SymbolList {
  using Tuple = SmallVector<Attribute, 3>;

  typedef enum { TerminalKind, SymbolTupleKind } Kind;

  /// Returns whether the symbol is of the appropriate class.
  static inline bool classof(SymbolList const *sym) { return true; }

  Kind getKind() const { return kind; }

  /// Returns a terminal uniqued in the context.
  static Terminal *get(SymbolContext &ctx, Attribute terminal);
  /// Returns a terminal uniqued in the context.
  static SymbolTuple *get(SymbolContext &ctx, Tuple &&tuple);
  /// Returns an attribute representation of the symbol.
  Attribute getAttr(MLIRContext *ctx) const;
  /// Computes the product of two symbol sets.
  static SymbolList *product(SymbolContext &ctx, SymbolList *lhs,
                             SymbolList *rhs, size_t k = 1);

protected:
  SymbolList(Kind kind) : kind(kind) {}

private:
  /// Symbol kind.
  Kind kind;
};

/// Symbol representing a terminal.
struct Terminal : SymbolList {
  friend struct SymbolContext;
  friend struct SymbolList;

  /// Returns an attribute representation of the symbol.
  Attribute getAttr(MLIRContext *ctx) const { return terminal; }

  /// Returns whether the symbol is of the appropriate class.
  static inline bool classof(SymbolList const *sym) {
    return sym->getKind() == TerminalKind;
  }

  /// Returns the terminal attribute.
  Attribute getTerminal() const { return terminal; }

  /// Compares two terminals.
  bool operator==(const Terminal &terminal) const {
    return this->terminal == terminal.terminal;
  }

  /// Hashes the terminal.
  llvm::hash_code hash_value() const {
    return llvm::hash_value(terminal.getAsOpaquePointer());
  }

  /// Returns the empty key.
  static Terminal getEmptyKey() {
    return Attribute::getFromOpaquePointer(
        ::llvm::DenseMapInfo<void *>::getEmptyKey());
  }

  /// Returns the tombstone key.
  static Terminal getTombstoneKey() {
    return Attribute::getFromOpaquePointer(
        ::llvm::DenseMapInfo<void *>::getTombstoneKey());
  }

private:
  Terminal(Attribute terminal) : SymbolList(TerminalKind), terminal(terminal) {}

  /// Terminal attribute.
  Attribute terminal;
};

/// Symbol representing a tuple of symbols.
struct SymbolTuple : SymbolList {
  friend struct SymbolContext;
  friend struct SymbolList;

  /// Returns an attribute representation of the symbol.
  Attribute getAttr(MLIRContext *ctx) const {
    return ArrayAttr::get(ctx, tuple);
  }

  /// Returns whether the symbol is of the appropriate class.
  static inline bool classof(SymbolList const *sym) {
    return sym->getKind() == SymbolTupleKind;
  }

  /// Compares two tuples.
  bool operator==(const SymbolTuple &tpl) const { return tuple == tpl.tuple; }

  /// Hashes the tuple.
  llvm::hash_code hash_value() const {
    return llvm::hash_value(ArrayRef<Attribute>(tuple));
  }

  /// Returns the empty key.
  static SymbolTuple getEmptyKey() { return SymbolTuple({}); }

  /// Returns the tombstone key.
  static SymbolTuple getTombstoneKey() {
    return SymbolTuple({Attribute::getFromOpaquePointer(
        ::llvm::DenseMapInfo<void *>::getTombstoneKey())});
  }

private:
  SymbolTuple(Tuple &&tuple)
      : SymbolList(SymbolTupleKind), tuple(std::move(tuple)) {}

  /// Symbol tuple.
  Tuple tuple;
};

/// Symbol context
struct SymbolContext {
  using Tuple = SymbolTuple::Tuple;
  /// Returns a pointer to a terminal uniqued in the context.
  Terminal *getTerminal(Attribute terminal);
  /// Returns a pointer to a tuple uniqued in the context.
  SymbolTuple *getTuple(Tuple &&tuple);

private:
  DenseMap<Attribute, std::unique_ptr<Terminal>> terminals;
  DenseMap<SymbolTuple, std::unique_ptr<SymbolTuple>> tuples;
};

/// A set of symbols.
struct SymbolSet {
  // we use a sorted set to maintain an stable order for computing intersections
  // in the PEGVisitor
  using Set = std::set<SymbolList *>;
  using Scratch = SmallVector<SymbolList *, 32>;
  SymbolSet() = default;

  SymbolSet(const Set &symSet) : symSet(symSet) {}

  /// Clones the set.
  SymbolSet clone() const { return SymbolSet(symSet); }

  /// Adds a terminal to the set, returns true if the terminal is new to the
  /// set.
  bool addSymbol(SymbolContext &ctx, Attribute terminal) {
    return symSet.insert(SymbolList::get(ctx, terminal)).second;
  }

  /// Adds the empty string, returns true if it is new to the set.
  bool addSymbol(SymbolContext &ctx, std::nullptr_t) {
    return symSet.insert(nullptr).second;
  }

  /// Returns whether the set is the null set.
  bool isNullSet() const {
    return symSet.empty() || (symSet.size() == 1 && isNullable());
  }

  /// Returns whether the set has the empty string.
  bool isNullable() const;
  /// Joins two sets, returns whether the set was modified or not.
  bool join(const SymbolSet &rhs);
  /// Computes the product of two sets, returns whether the set was modified or
  /// not.
  bool product(SymbolContext &ctx, const SymbolSet &lhs, const SymbolSet &rhs,
               size_t k = 1);
  /// Returns an attribute representation of the set.
  ArrayAttr getAttr(MLIRContext *ctx);
  Set symSet;
};

/// Class for holding the analysis of the grammar for each symbol.
struct SymbolInfo {
  /// Get the first attribute.
  ArrayAttr getFirstAttr(MLIRContext *ctx) { return firstSet.getAttr(ctx); }

  /// Get the follow attribute.
  ArrayAttr getFollowAttr(MLIRContext *ctx) { return followSet.getAttr(ctx); }

  SymbolSet firstSet;
  SymbolSet followSet;
  int visitCount = 0;
};

/// Generic grammar visitor.
template <typename Derived>
struct GenericVisitor {
  Derived &getDerived() { return static_cast<Derived &>(*this); }

  // Visits a switch operation.
  LogicalResult visitSwitch(SwitchOp op) {
    for (auto arg : op.getAlternatives())
      if (failed(visit(arg.getDefiningOp())))
        return failure();
    return success();
  }

  // Visits an any operation.
  LogicalResult visitAny(AnyOp op) {
    for (auto arg : op.getAlternatives())
      if (failed(visit(arg.getDefiningOp())))
        return failure();
    return success();
  }

  // Visits a seq operation.
  LogicalResult visitSequence(SequenceOp op) {
    for (auto arg : op.getAlternatives())
      if (failed(visit(arg.getDefiningOp())))
        return failure();
    return success();
  }

  LogicalResult visit(Operation *op) {
    assert(op && "invalid null operation");
    if (auto orOp = dyn_cast<OrOp>(op))
      return getDerived().visitOr(orOp);
    else if (auto andOp = dyn_cast<AndOp>(op))
      return getDerived().visitAnd(andOp);
    else if (auto zomOp = dyn_cast<ZeroOrMoreOp>(op))
      return getDerived().visitZeroOrMore(zomOp);
    else if (auto terminalOp = dyn_cast<TerminalOp>(op))
      return getDerived().visitTerminal(terminalOp);
    else if (auto ntOp = dyn_cast<EmptyStringOp>(op))
      return getDerived().visitEmptyString(ntOp);
    else if (auto ntOp = dyn_cast<NonTerminalOp>(op))
      return getDerived().visitNonTerminal(ntOp);
    else if (auto ntOp = dyn_cast<MetadataOp>(op))
      return getDerived().visitMetadata(ntOp);
    else if (auto swOp = dyn_cast<SwitchOp>(op))
      return getDerived().visitSwitch(swOp);
    else if (auto seqOp = dyn_cast<SequenceOp>(op))
      return getDerived().visitSequence(seqOp);
    else if (auto anyOp = dyn_cast<AnyOp>(op))
      return getDerived().visitAny(anyOp);
    return getDerived().visitPost(op);
  }
};

struct FirstVisitor;
struct PEGVisitor;

// Analyze the grammar.
class GrammarAnalysis {
public:
  using Symbol = void *;
  using TerminalSet = SmallVector<Attribute>;
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GrammarAnalysis);

  GrammarAnalysis(ParserOp op) {
    SymbolTable symTable(op);
    this->symTable = &symTable;
    emptyStr = UnitAttr::get(op.getContext());
    endOfFile = FlatSymbolRefAttr::get(op.getContext(), "EOF$");
    startSymbol = op.getStartSymbolAttr();
    analyze(op);
    this->symTable = nullptr;
  }

  SymbolInfo *getInfo(Symbol symbol) {
    auto it = info.find(symbol);
    if (it != info.end())
      return &(it->second);
    return nullptr;
  }

  void insertSymbolSet(Symbol) {}

private:
  friend struct FirstVisitor;
  friend struct PEGVisitor;
  // Analyzes the grammar.
  void analyze(ParserOp op);

  // Maps terminals to the possible following positions.
  DenseMap<Symbol, SymbolInfo> info;
  // Module's symbol table.
  SymbolTable *symTable;
  // Number of changes that occurred in an iteration.
  int changes = 0;
  // Attribute representing the empty string.
  UnitAttr emptyStr;
  // Attribute representing the end of file.
  FlatSymbolRefAttr endOfFile;
  // The parser start symbol.
  FlatSymbolRefAttr startSymbol;
  // Symbol context.
  SymbolContext symCtx;
  // Internal iteration counter.
  size_t iteration = 0;
};

/// First set visitor.
struct FirstVisitor : public GenericVisitor<FirstVisitor> {
  FirstVisitor(GrammarAnalysis &analysis) : analysis(analysis) {}

  // Visits a rule operation.
  LogicalResult visitRule(RuleOp rule);
  // Visits an or operation.
  LogicalResult visitOr(OrOp op);
  // Visits an and operation.
  LogicalResult visitAnd(AndOp op);
  // Visits a zero or more operation.
  LogicalResult visitZeroOrMore(ZeroOrMoreOp op);
  // Visits a terminal operation.
  LogicalResult visitTerminal(TerminalOp op);
  // Visits a terminal operation.
  LogicalResult visitEmptyString(EmptyStringOp op);
  // Visits a non-terminal operation.
  LogicalResult visitNonTerminal(NonTerminalOp op);
  // Visits a non-terminal operation.
  LogicalResult visitMetadata(MetadataOp op);

  LogicalResult visitPost(Operation *op) { return success(); }

  GrammarAnalysis &analysis;
};

/// Helper class for computing AnyOp nodes.
struct AnyNode {
  /// Returns the node value, either an AnyOp or a single value.
  Value getNode(OpBuilder &builder);
  /// Returns the first set.
  Attribute getFirstSet(OpBuilder &builder);
  SmallVector<Value> operands;
  SmallVector<Attribute> firstSets;
  SmallVector<Attribute> conflictSets;
  bool nullable = false;
};

/// PEG visitor.
struct PEGVisitor : public GenericVisitor<PEGVisitor> {
  PEGVisitor(MLIRContext &context, GrammarAnalysis &analysis)
      : context(context), analysis(analysis) {}

  // Visits a rule operation.
  LogicalResult visitRule(RuleOp rule);
  // Visits an or operation.
  LogicalResult visitOr(OrOp op);
  // Visits an and operation.
  LogicalResult visitAnd(AndOp op);
  // Visits a zero or more operation.
  LogicalResult visitZeroOrMore(ZeroOrMoreOp op);
  // Visits a terminal operation.
  LogicalResult visitTerminal(TerminalOp op);
  // Visits a terminal operation.
  LogicalResult visitEmptyString(EmptyStringOp op);
  // Visits a non-terminal operation.
  LogicalResult visitNonTerminal(NonTerminalOp op);
  // Visits a non-terminal operation.
  LogicalResult visitMetadata(MetadataOp op);
  // Rewrite the Or operation.
  Value rewriteOr(OrOp op, SmallVectorImpl<Value> &alternatives,
                  OpBuilder &builder);

  LogicalResult visitPost(Operation *op) { return success(); }

  MLIRContext &context;
  GrammarAnalysis &analysis;
};

struct AnalizeSyntax
    : public xblang::syntaxgen::impl::AnalyzeSyntaxBase<AnalizeSyntax> {
  using Base::Base;

  void runOnOperation() override;
};
} // namespace

//===----------------------------------------------------------------------===//
// SymbolContext
//===----------------------------------------------------------------------===//

namespace llvm {
using Terminal = ::Terminal;
using SymbolTuple = ::SymbolTuple;

template <>
struct DenseMapInfo<Terminal> {
  static inline Terminal getEmptyKey() { return Terminal::getEmptyKey(); }

  static inline Terminal getTombstoneKey() {
    return Terminal::getTombstoneKey();
  }

  static unsigned getHashValue(const Terminal &val) { return val.hash_value(); }

  static bool isEqual(const Terminal &lhs, const Terminal &rhs) {
    return lhs == rhs;
  }
};

template <>
struct DenseMapInfo<SymbolTuple> {
  static inline SymbolTuple getEmptyKey() { return SymbolTuple::getEmptyKey(); }

  static inline SymbolTuple getTombstoneKey() {
    return SymbolTuple::getTombstoneKey();
  }

  static unsigned getHashValue(const SymbolTuple &val) {
    return val.hash_value();
  }

  static bool isEqual(const SymbolTuple &lhs, const SymbolTuple &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

Terminal *SymbolContext::getTerminal(Attribute terminal) {
  auto &value = terminals[terminal];
  if (!value)
    value = std::unique_ptr<Terminal>(new Terminal(terminal));
  return value.get();
}

SymbolTuple *SymbolContext::getTuple(Tuple &&tuple) {
  auto sym = SymbolTuple(std::move(tuple));
  auto &value = tuples[sym];
  if (!value)
    value = std::unique_ptr<SymbolTuple>(new SymbolTuple(std::move(sym)));
  return value.get();
}

//===----------------------------------------------------------------------===//
// SymbolList
//===----------------------------------------------------------------------===//

Terminal *SymbolList::get(SymbolContext &ctx, Attribute terminal) {
  return ctx.getTerminal(terminal);
}

SymbolTuple *SymbolList::get(SymbolContext &ctx, Tuple &&tuple) {
  return ctx.getTuple(std::move(tuple));
}

Attribute SymbolList::getAttr(MLIRContext *ctx) const {
  if (kind == TerminalKind)
    return static_cast<const Terminal *>(this)->getAttr(ctx);
  else if (kind == SymbolTupleKind)
    return static_cast<const SymbolTuple *>(this)->getAttr(ctx);
  return nullptr;
}

SymbolList *SymbolList::product(SymbolContext &ctx, SymbolList *lhs,
                                SymbolList *rhs, size_t k) {
  // Return immediately if either side is the empty string (nullptr).
  if (!lhs)
    return rhs;
  if (!rhs)
    return lhs;
  auto sl = dyn_cast<SymbolTuple>(lhs);
  auto sr = dyn_cast<SymbolTuple>(rhs);
  auto tl = dyn_cast<Terminal>(lhs);
  auto tr = dyn_cast<Terminal>(rhs);
  // Return immediately if the LHS already has the appropriate size.
  if (sl && sl->tuple.size() >= k)
    return sl;
  if (tl && k == 1)
    return tl;
  // Create a new tuple with the LHS and RHS.
  if (sl && sr) {
    SmallVector<Attribute, 3> tpl(sl->tuple);
    auto &stpl = sr->tuple;
    size_t i = 0;
    while (tpl.size() < k && i < stpl.size())
      tpl.push_back(stpl[i++]);
    return get(ctx, std::move(tpl));
  }
  if (tl && tr)
    return get(ctx, {tl->getTerminal(), tr->getTerminal()});
  if (sl && tr) {
    SmallVector<Attribute, 3> tpl(sl->tuple);
    tpl.push_back(tr->getTerminal());
    return get(ctx, std::move(tpl));
  }
  if (tl && sr) {
    SmallVector<Attribute, 3> tpl({tl->getTerminal()});
    auto &stpl = sr->tuple;
    size_t i = 0;
    while (tpl.size() < k && i < stpl.size())
      tpl.push_back(stpl[i++]);
    return get(ctx, std::move(tpl));
  }
  llvm_unreachable("invalid lhs or rhs");
  return nullptr;
}

//===----------------------------------------------------------------------===//
// SymbolSet
//===----------------------------------------------------------------------===//

ArrayAttr SymbolSet::getAttr(MLIRContext *ctx) {
  SmallVector<Attribute> arr;
  if (symSet.empty())
    return nullptr;
  for (SymbolList *symList : symSet) {
    if (symList)
      arr.push_back(symList->getAttr(ctx));
    else
      arr.push_back(UnitAttr::get(ctx));
  }
  return ArrayAttr::get(ctx, arr);
}

bool SymbolSet::isNullable() const { return symSet.count(nullptr); }

bool SymbolSet::join(const SymbolSet &rhs) {
  size_t sz = symSet.size();
  symSet.insert(rhs.symSet.begin(), rhs.symSet.end());
  return sz != symSet.size();
}

bool SymbolSet::product(SymbolContext &ctx, const SymbolSet &lhs,
                        const SymbolSet &rhs, size_t k) {
  // Return immediately if the RHS is null.
  if (rhs.isNullSet())
    return join(lhs);
  // Return immediately if the LHS is null.
  if (lhs.isNullSet())
    return join(rhs);
  // Compute the full product of all elements.
  size_t sz = symSet.size();
  Scratch tmp;
  for (SymbolList *ll : lhs.symSet)
    for (SymbolList *rl : rhs.symSet)
      tmp.push_back(SymbolList::product(ctx, ll, rl, k));
  symSet.insert(tmp.begin(), tmp.end());
  return sz != symSet.size();
}

//===----------------------------------------------------------------------===//
// GrammarAnalysis
//===----------------------------------------------------------------------===//

void GrammarAnalysis::analyze(ParserOp op) {
  iteration = 0;
  FirstVisitor firstVisitor(*this);
  while (true) {
    changes = 0;
    for (auto rule : op.getBody(0)->getOps<RuleOp>())
      if (failed(firstVisitor.visitRule(rule)))
        return;
    ++iteration;
    if (changes == 0)
      break;
  }
}

//===----------------------------------------------------------------------===//
// FirstVisitor
//===----------------------------------------------------------------------===//

LogicalResult FirstVisitor::visitRule(RuleOp op) {
  ReturnOp ret = dyn_cast_or_null<ReturnOp>(op.getBody(0)->getTerminator());
  auto expr = ret.getExpr().getDefiningOp();
  if (failed(visit(expr)))
    return failure();
  // Initialize the info slot, otherwise DenseMap might invalidate the ref.
  if (analysis.iteration == 0) {
    analysis.info[expr].visitCount++;
    analysis.info[op.getOperation()].visitCount++;
  }
  auto &opInfo = analysis.info[op.getOperation()];
  auto &exprInfo = analysis.info[expr];
  if (opInfo.firstSet.join(exprInfo.firstSet))
    ++analysis.changes;
  return success();
}

LogicalResult FirstVisitor::visitOr(OrOp op) {
  auto lhs = op.getLHS().getDefiningOp();
  auto rhs = op.getRHS().getDefiningOp();
  if (failed(visit(lhs)))
    return failure();
  if (failed(visit(rhs)))
    return failure();
  // Initialize the info slot, otherwise DenseMap might invalidate the ref.
  if (analysis.iteration == 0) {
    analysis.info[lhs].visitCount++;
    analysis.info[rhs].visitCount++;
    analysis.info[op.getOperation()].visitCount++;
  }
  auto &set = analysis.info[op.getOperation()].firstSet;
  if (set.join(analysis.info[lhs].firstSet))
    ++analysis.changes;
  if (set.join(analysis.info[rhs].firstSet))
    ++analysis.changes;
  return success();
}

LogicalResult FirstVisitor::visitAnd(AndOp op) {
  auto lhs = op.getLHS().getDefiningOp();
  auto rhs = op.getRHS().getDefiningOp();
  if (failed(visit(lhs)))
    return failure();
  if (failed(visit(rhs)))
    return failure();
  // Initialize the info slot, otherwise DenseMap might invalidate the ref.
  if (analysis.iteration == 0) {
    analysis.info[lhs].visitCount++;
    analysis.info[rhs].visitCount++;
    analysis.info[op.getOperation()].visitCount++;
  }
  auto &opInfo = analysis.info[op.getOperation()];
  auto &lhsSet = analysis.info[lhs].firstSet;
  auto &rhsSet = analysis.info[rhs].firstSet;
  auto sz = opInfo.firstSet.symSet.size();
  (void)opInfo.firstSet.join(lhsSet);
  if (lhsSet.isNullable()) {
    (void)opInfo.firstSet.join(rhsSet);
    if (!rhsSet.isNullable())
      opInfo.firstSet.symSet.erase(nullptr);
  }
  if (sz != opInfo.firstSet.symSet.size())
    ++analysis.changes;
  return success();
}

LogicalResult FirstVisitor::visitZeroOrMore(ZeroOrMoreOp op) {
  auto expr = op.getExpr().getDefiningOp();
  if (failed(visit(expr)))
    return failure();
  // Initialize the info slot, otherwise DenseMap might invalidate the ref.
  if (analysis.iteration == 0) {
    analysis.info[expr].visitCount++;
    analysis.info[op.getOperation()].visitCount++;
  }
  auto &set = analysis.info[op.getOperation()].firstSet;
  if (set.join(analysis.info[expr].firstSet))
    ++analysis.changes;
  if (analysis.iteration > 0)
    return success();
  if (set.addSymbol(analysis.symCtx, nullptr))
    ++analysis.changes;
  return success();
}

LogicalResult FirstVisitor::visitTerminal(TerminalOp op) {
  if (analysis.iteration > 0)
    return success();
  if (analysis.info[op.getOperation()].firstSet.addSymbol(analysis.symCtx,
                                                          op.getTerminal()))
    ++analysis.changes;
  return success();
}

LogicalResult FirstVisitor::visitEmptyString(EmptyStringOp op) {
  if (analysis.iteration > 0)
    return success();
  if (analysis.info[op.getOperation()].firstSet.addSymbol(analysis.symCtx,
                                                          nullptr))
    ++analysis.changes;
  return success();
}

LogicalResult FirstVisitor::visitNonTerminal(NonTerminalOp op) {
  if (op.getNonTerminal() == "_$dyn") {
    if (analysis.iteration == 0) {
      analysis.info[op.getOperation()].visitCount++;
      analysis.info[op.getOperation()].firstSet.addSymbol(
          analysis.symCtx,
          LexTerminalAttr::get(op.getContext(), op.getNonTerminalAttr(),
                               LexTerminalKind::Any, *op.getDynamic()));
      ++analysis.changes;
    }
    return success();
  }
  auto nt = analysis.symTable->lookup(op.getNonTerminal());
  if (!nt)
    return emitError(op.getLoc(), "non-terminal can't be found");
  // Initialize the info slot, otherwise DenseMap might invalidate the ref.
  if (analysis.iteration == 0) {
    analysis.info[nt].visitCount++;
    analysis.info[op.getOperation()].visitCount++;
  }
  if (analysis.info[op.getOperation()].firstSet.join(
          analysis.info[nt].firstSet))
    ++analysis.changes;
  return success();
}

LogicalResult FirstVisitor::visitMetadata(MetadataOp op) {
  auto expr = op.getExpr().getDefiningOp();
  if (failed(visit(expr)))
    return failure();
  if (analysis.info[op.getOperation()].firstSet.join(
          analysis.info[expr].firstSet))
    ++analysis.changes;
  return success();
}

//===----------------------------------------------------------------------===//
// AnalizeSyntax
//===----------------------------------------------------------------------===//

void AnalizeSyntax::runOnOperation() {
  auto &analysis = getAnalysis<GrammarAnalysis>();
  ParserOp grammar = getOperation();
  bool addSet = false;
  if (addSet) {
    WalkResult result = grammar.walk([&](Operation *op) -> WalkResult {
      SymbolInfo *info = analysis.getInfo(op);
      if (!info || !addSet)
        return WalkResult::advance();
      if (Attribute attr = info->getFirstAttr(&getContext()))
        op->setAttr("first_set", attr);
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return signalPassFailure();
  }
  PEGVisitor visitor(getContext(), analysis);
  for (auto rule : grammar.getBody(0)->getOps<RuleOp>()) {
    if (failed(visitor.visitRule(rule)))
      return signalPassFailure();
  }
}

//===----------------------------------------------------------------------===//
// AnyNode
//===----------------------------------------------------------------------===//

Value AnyNode::getNode(OpBuilder &builder) {
  assert(operands.size() > 0 && "Invalid node.");
  if (operands.size() == 1)
    return operands.front();
  return builder.create<AnyOp>(operands.front().getLoc(), operands,
                               builder.getArrayAttr(firstSets),
                               builder.getArrayAttr(conflictSets), nullable);
}

Attribute AnyNode::getFirstSet(OpBuilder &builder) {
  assert(operands.size() > 0 && "Invalid node.");
  if (operands.size() == 1)
    return firstSets.front();
  SmallVector<Attribute, 32> scratch;
  for (auto attr : firstSets) {
    auto arr = cast<ArrayAttr>(attr);
    scratch.insert(scratch.end(), arr.getValue().begin(), arr.getValue().end());
  }
  auto less = [](mlir::Attribute x, mlir::Attribute y) {
    return x.getAsOpaquePointer() < y.getAsOpaquePointer();
  };
  std::stable_sort(scratch.begin(), scratch.end(), less);
  auto it = std::unique(scratch.begin(), scratch.end());
  scratch.erase(it, scratch.end());
  return builder.getArrayAttr(scratch);
}

//===----------------------------------------------------------------------===//
// PEGVisitor
//===----------------------------------------------------------------------===//

LogicalResult PEGVisitor::visitRule(RuleOp op) {
  ReturnOp ret = dyn_cast_or_null<ReturnOp>(op.getBody(0)->getTerminator());
  auto *firstSet = &analysis.info[op.getOperation()];
  if (auto attr = firstSet->getFirstAttr(&context))
    op->setAttr("first_set", attr);
  if (firstSet->firstSet.isNullable())
    op->setAttr("nullable", UnitAttr::get(&context));
  return visit(ret.getExpr().getDefiningOp());
}

Value PEGVisitor::rewriteOr(OrOp op, SmallVectorImpl<Value> &opts,
                            OpBuilder &builder) {
  auto less = [](mlir::Attribute x, mlir::Attribute y) {
    return x.getAsOpaquePointer() < y.getAsOpaquePointer();
  };
  SmallVector<std::pair<ArrayAttr, bool>, 8> optInfo(opts.size());
  llvm::IntEqClasses conflictClasses(opts.size());
  std::map<unsigned, AnyNode> operands;
  size_t sz = analysis.info.size(), maxElems = 0;
  // Get the first sets.
  for (size_t i = 0; i < opts.size(); ++i) {
    auto &info = analysis.info[opts[i].getDefiningOp()];
    optInfo[i] = {info.firstSet.getAttr(&context), info.firstSet.isNullable()};
    maxElems = std::max(maxElems, optInfo[i].first.size());
  }
  {
    SmallVector<SmallVector<Attribute>> conflicts(opts.size());
    SmallVector<Attribute, 32> scratch(maxElems);
    // Compute all possible conflicts.
    for (size_t i = 0; i < opts.size(); ++i) {
      for (size_t j = i + 1; j < opts.size(); ++j) {
        scratch.resize(maxElems);
        // Compute the set intersection.
        auto it = std::set_intersection(
            optInfo[i].first.begin(), optInfo[i].first.end(),
            optInfo[j].first.begin(), optInfo[j].first.end(), scratch.begin(),
            less);
        if ((it - scratch.begin()) > 0) {
          // Insert the conflicted terminal sequences.
          conflicts[i].insert(conflicts[i].end(), scratch.begin(), it);
          conflicts[j].insert(conflicts[j].end(), scratch.begin(), it);
          conflictClasses.join(i, j);
        }
        scratch.clear();
      }
      // Remove duplicate conflicts.
      std::sort(conflicts[i].begin(), conflicts[i].end(), less);
      auto it = std::unique(conflicts[i].begin(), conflicts[i].end());
      conflicts[i].erase(it, conflicts[i].end());
      AnyNode &node = operands[conflictClasses.findLeader(i)];
      node.operands.push_back(opts[i]);
      node.firstSets.push_back(optInfo[i].first);
      node.nullable = node.nullable || optInfo[i].second;
      node.conflictSets.push_back(builder.getArrayAttr(conflicts[i]));
      conflicts[i].clear();
    }
  }
  opts.clear();
  SmallVector<Attribute, 8> predictSets;
  bool nullable = false;
  for (auto &[k, v] : operands) {
    opts.push_back(v.getNode(builder));
    predictSets.push_back(v.getFirstSet(builder));
    nullable = nullable || v.nullable;
  }
  if (opts.size() == 1)
    return opts.front();
  (void)sz;
  assert(sz == analysis.info.size() &&
         "a key was added to the map invalidating the results");
  return builder.create<SwitchOp>(op.getLoc(), opts,
                                  builder.getArrayAttr(predictSets), nullable);
}

LogicalResult PEGVisitor::visitOr(OrOp op) {
  using pair_t = std::pair<Value, bool>;
  std::deque<pair_t> stack({pair_t{op, false}});
  SmallVector<Value, 8> values;
  SmallVector<OrOp, 8> eraseList;
  SymbolInfo info;
  { info = analysis.info[op.getOperation()]; }
  while (stack.size() > 0) {
    auto [val, color] = stack.front();
    stack.pop_front();
    OrOp orOp = dyn_cast<OrOp>(val.getDefiningOp());
    // If not an `or` op, promote the value
    if (!orOp) {
      values.push_back(val);
      continue;
    }
    if (!color) {
      stack.push_front({orOp, true});
      stack.push_front({orOp.getRHS(), false});
      stack.push_front({orOp.getLHS(), false});
    } else {
      eraseList.push_back(orOp);
    }
  }
  OpBuilder builder(op);
  auto rop = rewriteOr(op, values, builder);
  op->replaceAllUsesWith(ValueRange({rop}));
  for (auto op : llvm::reverse(eraseList))
    if (op->getUses().empty())
      op.erase();
  for (auto val : values)
    if (failed(visit(val.getDefiningOp())))
      return failure();
  analysis.info[rop.getDefiningOp()] = info;
  return success();
}

LogicalResult PEGVisitor::visitAnd(AndOp op) {
  using pair_t = std::pair<Value, bool>;
  std::deque<pair_t> stack({pair_t{op, false}});
  SmallVector<Value, 8> values;
  SmallVector<AndOp, 8> eraseList;
  SymbolInfo info;
  { info = analysis.info[op.getOperation()]; }
  LLVM_DEBUG({
    auto line = std::string(80, '*');
    llvm::dbgs() << line << "\nPromoting `and`: ";
    op.print(llvm::dbgs());
    llvm::dbgs() << "\n";
  });
  while (stack.size() > 0) {
    auto [val, color] = stack.front();
    stack.pop_front();
    AndOp andOp = dyn_cast<AndOp>(val.getDefiningOp());
    // If not an `and` op, promote the value
    if (!andOp) {
      values.push_back(val);
      LLVM_DEBUG({
        llvm::dbgs() << "    promoting value: ";
        val.print(llvm::dbgs());
        llvm::dbgs() << "\n";
      });
      continue;
    }
    LLVM_DEBUG({
      llvm::dbgs() << "  visiting[" << (color ? "last" : "first") << "]: ";
      andOp.print(llvm::dbgs());
      llvm::dbgs() << "\n";
    });
    if (!color) {
      stack.push_front({andOp, true});
      stack.push_front({andOp.getRHS(), false});
      stack.push_front({andOp.getLHS(), false});
    } else {
      eraseList.push_back(andOp);
    }
  }
  OpBuilder builder(op);
  auto seqOp = builder.create<SequenceOp>(op.getLoc(), values);
  LLVM_DEBUG({
    auto line = std::string(80, '*');
    llvm::dbgs() << "to `seq`: ";
    seqOp.print(llvm::dbgs());
    llvm::dbgs() << "\n" << line << "\n";
  });
  op->replaceAllUsesWith(seqOp);
  for (auto op : llvm::reverse(eraseList))
    if (op->getUses().empty())
      op.erase();
  for (auto val : values)
    if (failed(visit(val.getDefiningOp())))
      return failure();
  analysis.info[seqOp.getOperation()] = info;
  return success();
}

LogicalResult PEGVisitor::visitZeroOrMore(ZeroOrMoreOp op) {
  if (failed(visit(op.getExpr().getDefiningOp())))
    return failure();
  auto &info = analysis.info[op.getExpr().getDefiningOp()];
  op.setFirstSetAttr(info.getFirstAttr(&context));
  op.setNullable(info.firstSet.isNullable());
  return success();
}

LogicalResult PEGVisitor::visitTerminal(TerminalOp op) { return success(); }

LogicalResult PEGVisitor::visitEmptyString(EmptyStringOp op) {
  return success();
}

LogicalResult PEGVisitor::visitNonTerminal(NonTerminalOp op) {
  return success();
}

LogicalResult PEGVisitor::visitMetadata(MetadataOp op) {
  return visit(op.getExpr().getDefiningOp());
}
