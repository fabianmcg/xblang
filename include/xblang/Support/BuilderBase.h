//===- Builder.h - Builder utilities  ----------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares builder utilities.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_SUPPORT_BUILDERBASE_H
#define XBLANG_SUPPORT_BUILDERBASE_H

#include "mlir/IR/Builders.h"

namespace xblang {
/// RAII guard to reset the insertion point of the builder when destroyed.
class InsertionGuard {
public:
  using InsertPoint = mlir::OpBuilder::InsertPoint;
  InsertionGuard() = default;

  InsertionGuard(mlir::OpBuilder &builder)
      : builder(&builder), ip(builder.saveInsertionPoint()) {}

  ~InsertionGuard() { restore(); }

  InsertionGuard(const InsertionGuard &) = delete;
  InsertionGuard &operator=(const InsertionGuard &) = delete;

  /// Implement the move constructor to clear the builder field of `other`.
  InsertionGuard(InsertionGuard &&other) noexcept
      : builder(other.builder), ip(other.ip) {
    other.builder = nullptr;
  }

  InsertionGuard &operator=(InsertionGuard &&other) {
    builder = std::exchange(other.builder, nullptr);
    ip = std::exchange(other.ip, InsertPoint());
    return *this;
  }

  operator bool() const { return builder && ip.isSet(); }

  /// Restores the insertion point.
  void restore() {
    if (builder)
      builder->restoreInsertionPoint(ip);
    builder = nullptr;
  }

  /// Resets the insertion guard.
  InsertPoint reset(InsertPoint other = {}) {
    std::swap(other, ip);
    return other;
  }

  /// Abandons the insertion guard.
  InsertPoint abandon() {
    auto tmp = std::exchange(ip, InsertPoint());
    builder = nullptr;
    return tmp;
  }

  /// Returns the insertion point.
  InsertPoint getPoint() const { return ip; }

private:
  mlir::OpBuilder *builder = nullptr;
  InsertPoint ip = {};
};

/// RAII-styled insertion block, the object must be released to preserve the
/// block.
class InsertionBlock {
public:
  InsertionBlock() = default;
  ~InsertionBlock() = default;
  InsertionBlock(InsertionBlock &&) = default;
  InsertionBlock &operator=(InsertionBlock &&) = default;

  InsertionBlock(mlir::OpBuilder &builder)
      : insertionGuard(builder), block(new mlir::Block()) {
    builder.setInsertionPoint(block.get(), block->end());
  }

  operator bool() const { return block.get() != nullptr; }

  /// Releases the insertion block and restores the insertion point.
  mlir::Block *release() {
    insertionGuard.restore();
    return block.release();
  }

  /// Restores the insertion point.
  void restorePoint() { insertionGuard.restore(); }

  /// Abandons the insertion point.
  InsertionGuard::InsertPoint abandonPoint() {
    return insertionGuard.abandon();
  }

  /// Returns the held block.
  mlir::Block *getBlock() { return block.get(); }

private:
  InsertionGuard insertionGuard{};
  std::unique_ptr<mlir::Block> block{};
};

struct BuilderBase {
  using Builder = mlir::OpBuilder;
  using Guard = InsertionGuard;

  /// Creates a builder guard.
  static Guard guard(Builder &builder) { return Guard(builder); }

  /// Creates a builder guard and sets the insertion point.
  static Guard guard(Builder &builder, mlir::Operation *op) {
    Guard guard(builder);
    builder.setInsertionPoint(op);
    return guard;
  }

  static Guard guard(Builder &builder, mlir::Block *block,
                     mlir::Block::iterator point) {
    Guard guard(builder);
    builder.setInsertionPoint(block, point);
    return guard;
  }

  /// Creates a builder guard and sets the insertion point to the end of the
  /// block.
  static Guard guard(Builder &builder, mlir::Block *block) {
    Guard guard(builder);
    builder.setInsertionPoint(block, block->end());
    return guard;
  }

  /// Creates a builder guard and sets the insertion point after.
  static Guard guardAfter(Builder &builder, mlir::Operation *op) {
    Guard guard(builder);
    builder.setInsertionPointAfter(op);
    return guard;
  }

  static Guard guardAfter(Builder &builder, mlir::Value value) {
    Guard guard(builder);
    builder.setInsertionPointAfterValue(value);
    return guard;
  }
};
} // namespace xblang

#endif // XBLANG_SUPPORT_BUILDERBASE_H
