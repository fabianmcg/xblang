//===- Worklist.h - Wokrlist for traversing the IR ---------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares a work-list utility class for traversing the IR.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_SUPPORT_WORKLIST_H
#define XBLANG_SUPPORT_WORKLIST_H

#include "mlir/IR/Operation.h"
#include "xblang/Support/LLVM.h"
#include <deque>

namespace xblang {
/// Class for holding elements of a work-list including metadata.
class alignas(8) IRWorklistElementStatus {
public:
  IRWorklistElementStatus() = default;

  /// Returns the number of times the element has been visited.
  uint32_t getCount() const { return visitCount; }

  /// Increments the visitor counter.
  IRWorklistElementStatus &operator++() {
    if (visitCount < std::numeric_limits<int32_t>::max())
      visitCount++;
    return *this;
  }

  /// Returns the internal state.
  uint32_t &getState() { return state; }

  uint32_t getState() const { return state; }

protected:
  /// Number of times the element has been visited.
  uint32_t visitCount = 0;
  /// Internal state.
  uint32_t state = 0;
};

/// Class for holding elements of a work-list including metadata.
template <typename IRWorklistElementStatus>
class GenericIRWorklistElement : public IRWorklistElementStatus {
public:
  GenericIRWorklistElement() = default;

  GenericIRWorklistElement(mlir::Operation *op,
                           IRWorklistElementStatus status = {})
      : IRWorklistElementStatus(status), element(op) {}

  operator bool() const { return element; }

  /// Increments the visitor counter.
  GenericIRWorklistElement &operator++() {
    IRWorklistElementStatus::operator++();
    return *this;
  }

  /// Returns the held operation.
  mlir::Operation *get() { return element; }

private:
  /// Element being held.
  mlir::Operation *element = nullptr;
};

/// Class for holding an IR work-list.
template <template <typename...> class Container, typename IRWorklistElement>
class GenericIRWorklist : public Container<IRWorklistElement> {
public:
  using Base = Container<IRWorklistElement>;
  using Base::Base;

  /// Returns the top element in the work-list.
  IRWorklistElement top() {
    if (this->size())
      return this->back();
    return {};
  }

  /// Adds the operands of an operation to the work-list.
  size_t addOperands(mlir::Operation *op) {
    if (!op)
      return 0;
    size_t c = 0;
    for (auto operand : llvm::reverse(op->getOperands()))
      c += addValue(operand);
    return c;
  }

  /// Adds the ops in the regions of an operation to the work-list.
  size_t addRegions(mlir::Operation *op) {
    if (!op)
      return 0;
    size_t c = 0;
    for (mlir::Region &region : llvm::reverse(op->getRegions()))
      for (mlir::Block &block : llvm::reverse(region))
        for (mlir::Operation &op : llvm::reverse(block))
          this->push_back(&op), c++;
    return c;
  }

  /// Adds the contents of an operation to the work-list.
  size_t addOp(mlir::Operation *op, bool insertOperands = false) {
    if (!op)
      return 0;
    size_t c = addRegions(op);
    if (insertOperands)
      c += addOperands(op);
    return c;
  }

  /// Adds a value to the work-list.
  size_t addValue(mlir::Value val, bool addOperands = false) {
    if (!val)
      return 0;
    return addOp(val.getDefiningOp(), addOperands);
  }

  /// Pops an element from the back of the work-list.
  IRWorklistElement pop(bool increment = true) {
    IRWorklistElement elem{};
    if (this->size()) {
      elem = this->back();
      if (increment)
        ++elem;
      this->pop_back();
    }
    return elem;
  }
};

template <typename T>
using IRWorklistSmallVec = llvm::SmallVector<T>;

using IRWorklistElement = GenericIRWorklistElement<IRWorklistElementStatus>;
using IRWorklist = GenericIRWorklist<IRWorklistSmallVec, IRWorklistElement>;
} // namespace xblang

#endif // XBLANG_SUPPORT_WORKLIST_H
