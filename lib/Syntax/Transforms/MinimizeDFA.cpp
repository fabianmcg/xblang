//===- MinimizeDFA.cpp - Minimize DFA pass -----------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the minimize DFA pass.
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
#include "llvm/ADT/IntEqClasses.h"
#include "llvm/ADT/SmallSet.h"
#include <compare>
#include <list>

namespace xblang {
namespace syntaxgen {
#define GEN_PASS_DEF_MINIMIZEDFA
#include "xblang/Syntax/Transforms/Passes.h.inc"
} // namespace syntaxgen
} // namespace xblang

using namespace mlir;
using namespace xblang;
using namespace xblang::syntaxgen;

namespace {
// Minimizer pass
struct MinimizeDFA
    : public xblang::syntaxgen::impl::MinimizeDFABase<MinimizeDFA> {
  using Base::Base;

  void runOnOperation() override;
};

//===----------------------------------------------------------------------===//
// FiniteAutomata
//===----------------------------------------------------------------------===//
/// Finite automata in a mixed CSR & COO form.
class FiniteAutomata {
public:
  using State = uint32_t;
  using Symbol = void *;

  struct Transition {
    State source;
    State target;
    Symbol symbol;
    std::strong_ordering operator<=>(const Transition &other) const;
  };

  using TransitionFunction = SmallVector<Transition>;
  using iterator = Transition *;
  using TransitionRange = std::pair<iterator, iterator>;

  /// Adds a new state.
  State addState() { return numberOfStates++; }

  /// Adds a new state transition.
  void addTransition(State source, State target, Symbol sym) {
    transitionFunction.push_back({source, target, sym});
  }

  /// Adds a final state to the FA.
  void addFinalState(State state) {
    assert(state < numberOfStates && "Invalid state ID.");
    finalStates.insert(state);
  }

  // Returns the number of states in the FA.
  size_t size() const { return numberOfStates; }

  /// Returns iterators for the transitions starting from source.
  std::pair<iterator, iterator> transitions(State source);
  /// Compresses the automata.
  void compress();

  /// Returns the set of final states.
  const llvm::SmallSet<State, 32> &getFinalStates() const {
    return finalStates;
  }

protected:
  TransitionFunction transitionFunction;
  SmallVector<size_t> transitionPtr;
  llvm::SmallSet<State, 32> finalStates;
  size_t numberOfStates = 0;
};

//===----------------------------------------------------------------------===//
// AutomataPartitioner
//===----------------------------------------------------------------------===//
class AutomataPartitioner {
public:
  using State = FiniteAutomata::State;
  using Transition = FiniteAutomata::Transition;
  using TransitionRange = FiniteAutomata::TransitionRange;
  using iterator = FiniteAutomata::iterator;
  using PID = int; // Partition ID.
  using PartPtr = State;

  /// Computes the state equivalence classes, returns empty if the DFA is
  /// already minimal.
  static llvm::IntEqClasses getClasses(FiniteAutomata &automata) {
    return AutomataPartitioner(automata).computeClasses();
  }

protected:
  AutomataPartitioner(FiniteAutomata &automata) : automata(automata) { init(); }

  /// Computes the equivalence classes fot the FA.
  llvm::IntEqClasses computeClasses();
  /// Compares the transitions inside of two states.
  bool compareTransitions(TransitionRange, TransitionRange);
  void init();
  FiniteAutomata &automata;
  SmallVector<PID> partition;
  SmallVector<PID> partitionScratch;
  SmallVector<State> permutation;
  std::list<std::pair<PartPtr, PartPtr>> partitionPtr;
};
} // namespace

//===----------------------------------------------------------------------===//
// FiniteAutomata
//===----------------------------------------------------------------------===//

std::strong_ordering
FiniteAutomata::Transition::operator<=>(const Transition &other) const {
  if (source < other.source)
    return std::strong_ordering::less;
  auto tmp = source == other.source;
  if (tmp && (symbol < other.symbol))
    return std::strong_ordering::less;
  tmp = tmp && (symbol == other.symbol);
  if (tmp && (target == other.target))
    return std::strong_ordering::equivalent;
  return std::strong_ordering::greater;
}

FiniteAutomata::TransitionRange FiniteAutomata::transitions(State state) {
  if (auto ptr = transitionFunction.data(); state < numberOfStates)
    return {ptr + transitionPtr[state], ptr + transitionPtr[state + 1]};
  return {nullptr, nullptr};
}

void FiniteAutomata::compress() {
  std::sort(transitionFunction.begin(), transitionFunction.end(),
            [](const Transition &t1, const Transition &t2) {
              return (t1 <=> t2) < 0;
            });
  auto last = std::unique(transitionFunction.begin(), transitionFunction.end(),
                          [](const Transition &t1, const Transition &t2) {
                            return (t1 <=> t2) == 0;
                          });
  transitionFunction.erase(last, transitionFunction.end());
  transitionPtr.resize(numberOfStates + 1);
  for (Transition &transition : transitionFunction)
    transitionPtr[transition.source + 1]++;
  for (size_t i = 1; i < transitionPtr.size(); ++i)
    transitionPtr[i] += transitionPtr[i - 1];
}

//===----------------------------------------------------------------------===//
// AutomataPartitioner
//===----------------------------------------------------------------------===//

void AutomataPartitioner::init() {
  partition.resize(automata.size());
  partitionScratch.resize(automata.size());
  permutation.resize(automata.size());
  // Initialize with the identity permutation.
  for (size_t i = 0; i < partition.size(); ++i) {
    partition[i] = 0;
    permutation[i] = i;
  }
  // Move the final states to their own unique partition.
  PID pid = 1;
  for (State s : automata.getFinalStates())
    partition[s] = pid++;
  // Sort the permutation using the partition.
  std::sort(permutation.begin(), permutation.end(), [this](State s1, State s2) {
    return partition[s1] < partition[s2];
  });
  // Initialize the partition ptr with the two initial partitions.
  PartPtr ptr = 0;
  PID currentPartition = -1;
  auto pit = partitionPtr.begin();
  for (State piState : permutation) {
    PID p = partition[piState];
    partitionScratch[piState] = p;
    if (p != currentPartition) {
      currentPartition = p;
      pit = partitionPtr.insert(partitionPtr.end(), {ptr, ptr + 1});
      ++ptr;
      continue;
    }
    pit->second = ++ptr;
  }
}

llvm::IntEqClasses AutomataPartitioner::computeClasses() {
  // Sort the permutation using the partition.
  auto sort = [this](int b, int e) {
    std::sort(permutation.begin() + b, permutation.begin() + e,
              [this](int p1, int p2) { return partition[p1] < partition[p2]; });
  };
  // Fix-point algorithm for computing the partitions. It's guaranteed to
  // converge in a max of `number of states` steps.
  size_t numPartitions = partitionPtr.size(),
         prevNumPartitions = partitionPtr.size();
  for (size_t i = 0; i < automata.size(); ++i) {
    // Iterate through the partitions.
    auto it = partitionPtr.begin();
    while (it != partitionPtr.end()) {
      auto partIt = it++;
      PartPtr partBegin = partIt->first;
      PartPtr partEnd = partIt->second;
      // Continue if the partition has a single member.
      if (partBegin + 1 == partEnd)
        continue;
      size_t numPartCheckpoint = numPartitions;
      // Iterate through the partition members in a CSR sparse fashion.
      for (PartPtr ptr = partBegin; ptr < partEnd; ++ptr) {
        State state = permutation[ptr];
        PID part = partition[state];
        TransitionRange stateTranstitions = automata.transitions(state);
        // Compare against all the other members in the partition.
        for (PartPtr otherPtr = ptr + 1; otherPtr < partEnd; ++otherPtr) {
          State otherState = permutation[otherPtr];
          PID &otherPart = partition[otherState];
          // Continue if no comparison is needed.
          // If `otherPart == part`, they need to be compared as they could be
          // in different classes. If `otherPart > part`, they need to be
          // compared as they could be in the same new class.
          if (otherPart < part)
            continue;
          // Compare the transitions of tgt vs other, splitting the states if
          // they are not the same.
          if (compareTransitions(stateTranstitions,
                                 automata.transitions(otherState)))
            otherPart = part;
          else if (otherPart == part)
            otherPart = ++numPartitions;
        }
      }
      // Continue if no new partitions where detected.
      if (numPartitions == numPartCheckpoint)
        continue;
      // Sort the permutation according to the new partition.
      sort(partBegin, partEnd);
      // Insert the new partitions to the partition ptr.
      PID currentPart = -1;
      PartPtr ptr = partBegin;
      auto tmpIt = partIt;
      for (PartPtr partPtr = partBegin; partPtr < partEnd; ++partPtr) {
        State state = permutation[partPtr];
        PID p = partition[state];
        partitionScratch[state] = p;
        if (p != currentPart) {
          currentPart = p;
          tmpIt = partitionPtr.insert(++tmpIt, {ptr, ptr + 1});
          ++ptr;
          continue;
        }
        tmpIt->second = ++ptr;
      }
      // Dissolve the split partition.
      partitionPtr.erase(partIt);
    }
    // Check for convergence, ie. no new partitions were added.
    if (numPartitions == prevNumPartitions)
      break;
    prevNumPartitions = numPartitions;
  }
  // Return immediately if the FA is already minimal.
  if (partitionPtr.size() == automata.size())
    return {};
  // Compute the equivalence classes.
  llvm::IntEqClasses states(partition.size());
  for (auto &partitionMembers : partitionPtr) {
    auto classRepr = permutation[partitionMembers.first];
    for (State state = partitionMembers.first + 1;
         state < partitionMembers.second; ++state)
      classRepr = states.join(classRepr, permutation[state]);
  }
  states.compress();
  return states;
}

bool AutomataPartitioner::compareTransitions(TransitionRange t1,
                                             TransitionRange t2) {
  ptrdiff_t size = t1.second - t1.first;
  if (size != (t2.second - t2.first))
    return false;
  for (ptrdiff_t i = 0; i < size; ++i) {
    const Transition &tt1 = *(t1.first + i);
    const Transition &tt2 = *(t2.first + i);
    if (tt1.symbol != tt2.symbol)
      return false;
    if (partitionScratch[tt1.target] != partitionScratch[tt2.target])
      return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// MinimizeDFA
//===----------------------------------------------------------------------===//

void MinimizeDFA::runOnOperation() {
  auto grammar = getOperation();
  using State = FiniteAutomata::State;
  DenseMap<StringAttr, State> opToId;
  FiniteAutomata fa;
  SmallVector<LexStateOp> states;
  // Build the FA.
  for (auto state : grammar.getBody()->getOps<LexStateOp>()) {
    State stateId = fa.addState();
    opToId[state.getSymNameAttr()] = stateId;
    if (state.getFinalState())
      fa.addFinalState(stateId);
    states.push_back(state);
  }
  for (auto state : states) {
    if (state.getBodyRegion().empty())
      continue;
    State id = opToId[state.getSymNameAttr()];
    for (auto transition : state.getBody()->getOps<LexTransitionOp>()) {
      auto it = opToId.find(transition.getNextStateAttr().getAttr());
      if (it == opToId.end()) {
        transition.emitError()
            << "transition doesn't point to a valid next state";
        return signalPassFailure();
      }
      fa.addTransition(id, it->second,
                       transition.getTerminal().getAsOpaquePointer());
    }
  }
  fa.compress();
  // Compute the equivalence classes of states.
  llvm::IntEqClasses classes = AutomataPartitioner::getClasses(fa);
  if (classes.getNumClasses() == 0)
    return;
  // Update uniqued states and remove duplicated states.
  llvm::DenseSet<State> emitted;
  for (auto state : states) {
    auto stateClass = classes[opToId[state.getSymNameAttr()]];
    if (emitted.count(stateClass)) {
      state.erase();
      continue;
    }
    emitted.insert(stateClass);
    state.setName(fmt("State{0}{1}", grammar.getName(), stateClass));
    if (state.getBodyRegion().empty())
      continue;
    for (auto transition : state.getBody()->getOps<LexTransitionOp>())
      transition.setNextState(
          fmt("State{0}{1}", grammar.getName(),
              classes[opToId[transition.getNextStateAttr().getAttr()]]));
  }
}
