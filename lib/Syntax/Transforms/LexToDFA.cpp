//===- LexToDFA.cpp - SLex to DFA pass ---------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the lex to DFA pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xblang/Support/Format.h"
#include "xblang/Syntax/IR/SyntaxDialect.h"
#include "xblang/Syntax/Transforms/Passes.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include <deque>
#include <set>

namespace xblang {
namespace syntaxgen {
#define GEN_PASS_DEF_LEXTODFA
#include "xblang/Syntax/Transforms/Passes.h.inc"
} // namespace syntaxgen
} // namespace xblang

using namespace mlir;
using namespace xblang;
using namespace xblang::syntaxgen;

namespace {
using Symbol = Operation *;
using SymbolID = int32_t;
using SymbolSet = SmallVector<SymbolID>;

// Lex to DFA pass.
struct LexToDFA : public xblang::syntaxgen::impl::LexToDFABase<LexToDFA> {
  using Base::Base;

  void runOnOperation() override;
};

// Erases irrelevant lex operations.
struct LexOpEraserPattern : public RewritePattern {
  LexOpEraserPattern(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;
};

// Uniques the elements in the set.
SymbolSet &uniqueSet(SymbolSet &set) {
  std::sort(set.begin(), set.end());
  set.erase(std::unique(set.begin(), set.end()), set.end());
  return set;
}

// Computes the union of two sets.
template <typename T, typename V>
void setUnion(T &result, const V &rhs) {
  result.insert(result.end(), rhs.begin(), rhs.end());
}

// Analyze the grammar to determine followpos.
class GrammarAnalysis {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GrammarAnalysis);

  GrammarAnalysis(DFAOp op) { analyze(op); }

  // Returns the follow set for the respective symbol.
  const SymbolSet *getFollowpos(SymbolID id) const {
    auto it = followpos.find(id);
    if (it != followpos.end())
      return &(it->second);
    return nullptr;
  }

  // Returns the respective symbol.
  Symbol getSymbol(SymbolID id) const {
    auto it = lut.find(id);
    if (it != lut.end())
      return (it->second);
    return nullptr;
  }

  // Returns the entry state.
  const SymbolSet &getEntry() const { return entry; }

private:
  // Analyzes the grammar.
  void analyze(DFAOp op);
  // Visits an operation, returns whether the operation is nullable.
  bool visit(Operation *v, SymbolSet &firstpos, SymbolSet &lastpos);
  // Visits a rule operation, returns whether the operation is nullable.
  bool visitRule(LexRuleOp rule, SymbolSet &firstpos, SymbolSet &lastpos);
  // Visits an or operation, returns whether the operation is nullable.
  bool visitOr(Operation *lhs, Operation *rhs, SymbolSet &firstpos,
               SymbolSet &lastpos);
  bool visitOr(OrOp op, SymbolSet &firstpos, SymbolSet &lastpos);
  // Visits an and operation, returns whether the operation is nullable.
  bool visitAnd(Operation *lhs, Operation *rhs, SymbolSet &firstpos,
                SymbolSet &lastpos);
  bool visitAnd(AndOp op, SymbolSet &firstpos, SymbolSet &lastpos);
  // Visits a zero or more operation, returns true.
  bool visitZeroOrMore(ZeroOrMoreOp op, SymbolSet &firstpos,
                       SymbolSet &lastpos);
  // Visits a terminal operation, returns false.
  bool visitTerminal(TerminalOp op, SymbolSet &firstpos, SymbolSet &lastpos);

  // Computes the union of two sets.
  void setUnion(SymbolSet &set, const SymbolSet &lhs, const SymbolSet &rhs) {
    set.insert(set.end(), lhs.begin(), lhs.end());
    set.insert(set.end(), rhs.begin(), rhs.end());
  }

  // Utility function clearing the container before being used.
  template <typename T>
  static T &clear(T &container) {
    container.clear();
    return container;
  }

  // Entry point for the grammar.
  SymbolSet entry;
  // Maps terminals to the possible following positions.
  DenseMap<SymbolID, SymbolSet> followpos;
  // Maps terminals to the possible following positions.
  DenseMap<SymbolID, Symbol> lut;
};

// Deterministic DFA state.
struct DFAState {
  DFAState(const SymbolSet &state, int id) : state(state), id(id) {
    uniqueSet(this->state);
  }

  bool operator<(const DFAState &other) const { return state < other.state; }

  SymbolSet state;
  int id;
};

// Automata states container.
struct StateUniquer {
  // Creates or returns an unique state.
  std::pair<const DFAState *, bool> getUnique(const SymbolSet &state);
  // TerminalSet list.
  std::set<DFAState> states;
};
} // namespace

//===----------------------------------------------------------------------===//
// GrammarAnalysis
//===----------------------------------------------------------------------===//

void GrammarAnalysis::analyze(DFAOp grammar) {
  SymbolSet firstpos, lastpos;
  for (auto rule : grammar.getBodyRegion().getOps<LexRuleOp>()) {
    visitAnd(rule.getExpr().getDefiningOp(), rule, firstpos, lastpos);
    ::setUnion(entry, firstpos);
  }
  uniqueSet(entry);
  // Unique the terminals.
  for (auto &[k, v] : followpos)
    uniqueSet(v);
}

bool GrammarAnalysis::visitRule(LexRuleOp rule, SymbolSet &firstpos,
                                SymbolSet &lastpos) {
  SymbolID id = lut.size();
  lut[id] = rule;
  firstpos.push_back(id);
  lastpos.push_back(id);
  return false;
}

bool GrammarAnalysis::visitOr(Operation *lhs, Operation *rhs,
                              SymbolSet &firstpos, SymbolSet &lastpos) {
  SymbolSet lhsFirstpos, lhsLastpos, rhsFirstpos, rhsLastpos;
  bool lhsNullable = visit(lhs, lhsFirstpos, lhsLastpos);
  bool rhsNullable = visit(rhs, rhsFirstpos, rhsLastpos);
  ::setUnion(lhsFirstpos, rhsFirstpos);
  ::setUnion(lhsLastpos, rhsLastpos);
  firstpos = std::move(lhsFirstpos);
  lastpos = std::move(lhsLastpos);
  return lhsNullable || rhsNullable;
}

bool GrammarAnalysis::visitOr(OrOp op, SymbolSet &firstpos,
                              SymbolSet &lastpos) {
  return visitOr(op.getLHS().getDefiningOp(), op.getRHS().getDefiningOp(),
                 firstpos, lastpos);
}

bool GrammarAnalysis::visitAnd(Operation *lhs, Operation *rhs,
                               SymbolSet &firstpos, SymbolSet &lastpos) {
  SymbolSet lhsFirstpos, lhsLastpos, rhsFirstpos, rhsLastpos, localLastpos;
  bool lhsNullable = visit(lhs, lhsFirstpos, lhsLastpos);
  bool rhsNullable = visit(rhs, rhsFirstpos, rhsLastpos);
  if (lhsNullable)
    ::setUnion(lhsFirstpos, rhsFirstpos);
  if (rhsNullable)
    setUnion(localLastpos, lhsLastpos, rhsLastpos);
  else
    ::setUnion(localLastpos, rhsLastpos);
  for (auto symbol : lhsLastpos)
    ::setUnion(followpos[symbol], rhsFirstpos);
  firstpos = std::move(lhsFirstpos);
  lastpos = localLastpos;
  return lhsNullable && rhsNullable;
}

bool GrammarAnalysis::visitAnd(AndOp op, SymbolSet &firstpos,
                               SymbolSet &lastpos) {
  return visitAnd(op.getLHS().getDefiningOp(), op.getRHS().getDefiningOp(),
                  firstpos, lastpos);
}

bool GrammarAnalysis::visitZeroOrMore(ZeroOrMoreOp op, SymbolSet &firstpos,
                                      SymbolSet &lastpos) {
  visit(op.getExpr().getDefiningOp(), firstpos, lastpos);
  for (auto symbol : lastpos)
    ::setUnion(followpos[symbol], firstpos);
  return true;
}

bool GrammarAnalysis::visitTerminal(TerminalOp op, SymbolSet &firstpos,
                                    SymbolSet &lastpos) {
  SymbolID id = lut.size();
  lut[id] = op;
  firstpos.push_back(id);
  lastpos.push_back(id);
  return false;
}

bool GrammarAnalysis::visit(Operation *op, SymbolSet &firstpos,
                            SymbolSet &lastpos) {
  clear(firstpos);
  clear(lastpos);
  // op is likely to be null if it comes from a block argument.
  assert(op && "invalid null operation");
  if (auto orOp = dyn_cast<OrOp>(op))
    return visitOr(orOp, firstpos, lastpos);
  else if (auto andOp = dyn_cast<AndOp>(op))
    return visitAnd(andOp, firstpos, lastpos);
  else if (auto zomOp = dyn_cast<ZeroOrMoreOp>(op))
    return visitZeroOrMore(zomOp, firstpos, lastpos);
  else if (auto terminalOp = dyn_cast<TerminalOp>(op))
    return visitTerminal(terminalOp, firstpos, lastpos);
  else if (auto ruleOp = dyn_cast<LexRuleOp>(op))
    return visitRule(ruleOp, firstpos, lastpos);
  return true;
}

//===----------------------------------------------------------------------===//
// StateUniquer
//===----------------------------------------------------------------------===//

std::pair<const DFAState *, bool>
StateUniquer::getUnique(const SymbolSet &state) {
  auto result = states.insert(DFAState(state, states.size()));
  return {&(*result.first), result.second};
}

//===----------------------------------------------------------------------===//
// LexToDFA
//===----------------------------------------------------------------------===//

void LexToDFA::runOnOperation() {
  auto &analysis = getAnalysis<GrammarAnalysis>();
  auto grammar = getOperation();
  auto loc = grammar.getLoc();
  StateUniquer uniquer;
  OpBuilder builder(getOperation().getBody(), getOperation().getBody()->end());
  std::deque<const DFAState *> unmarked;
  // Insert the entry state.
  unmarked.push_back(uniquer.getUnique(analysis.getEntry()).first);
  // Deterministic transition map.
  llvm::MapVector<Symbol, SymbolSet> transitions;
  // Traverse all transitions creating the appropriate in the process.
  while (unmarked.size()) {
    // Consume an state.
    const DFAState *top = unmarked.front();
    unmarked.pop_front();
    // Check whether the state is a final state, ie. a rule.
    Operation *sym = nullptr;
    bool isFinal = (top->state.size() == 1) &&
                   isa<LexRuleOp>(sym = analysis.getSymbol(top->state[0]));
    StringAttr id =
        isFinal && sym ? dyn_cast<LexRuleOp>(sym).getNameAttr() : StringAttr();
    // Create the state operation
    auto stateOp = builder.create<LexStateOp>(
        loc, fmt("State{0}{1}", grammar.getName(), top->id), isFinal, id);
    if (isFinal)
      continue;
    // Compute all the transitions originating from the consumed state.
    transitions.clear();
    SmallVector<std::pair<Symbol, SymbolID>> epsTransitions;
    for (SymbolID symbolId : top->state) {
      Symbol symbol = analysis.getSymbol(symbolId);
      if (auto followpos = analysis.getFollowpos(symbolId))
        setUnion(transitions[symbol], *followpos);
      else if (auto rule = dyn_cast<LexRuleOp>(symbol))
        epsTransitions.push_back({symbol, symbolId});
    }
    // Make sure the epsilon transitions are at the end.
    for (auto &eps : epsTransitions)
      transitions[eps.first] = {eps.second};
    // Add a block to add all the transitions.
    Block *stateBlock =
        transitions.empty() ? nullptr : &stateOp.getBodyRegion().emplaceBlock();
    // Create the transitions.
    for (auto &[k, v] : transitions) {
      auto state = uniquer.getUnique(v);
      if (state.second)
        unmarked.push_back(state.first);
      {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToEnd(stateBlock);
        std::string name =
            fmt("State{0}{1}", grammar.getName(), state.first->id);
        if (auto temrinal = dyn_cast<TerminalOp>(k))
          builder.create<LexTransitionOp>(loc, temrinal.getResult(), name);
        else if (auto temrinal = dyn_cast<LexRuleOp>(k))
          builder.create<LexTransitionOp>(
              loc, builder.createOrFold<EmptyStringOp>(loc), name);
      }
    }
    if (epsTransitions.size() > 1)
      stateOp.emitWarning("The FA is not deterministic.");
  }
  // Erase all `and`, `or` and `sero_or_more` ops.
  RewritePatternSet patterns(&getContext());
  patterns.add<LexOpEraserPattern>(&getContext());
  if (failed(applyPatternsAndFoldGreedily(grammar, std::move(patterns))))
    return signalPassFailure();
}

LogicalResult
LexOpEraserPattern::matchAndRewrite(Operation *op,
                                    PatternRewriter &rewriter) const {
  if (isa<LexRuleOp>(op) || isa<AndOp>(op) || isa<ZeroOrMoreOp>(op)) {
    rewriter.eraseOp(op);
    return success();
  }
  return failure();
}
