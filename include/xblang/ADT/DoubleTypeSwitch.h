#ifndef XBLANG_ADT_DOUBLETYPESWITCH_H
#define XBLANG_ADT_DOUBLETYPESWITCH_H

#include "llvm/Support/Casting.h"
#include <optional>

namespace xblang {
template <typename U, typename V, typename Result>
class DoubleTypeSwitch {
private:
  const U *u{};
  const V *v{};
  std::optional<Result> result{};

  DoubleTypeSwitch(const U &u, const V &v) : u(&u), v(&v) {}

  DoubleTypeSwitch(DoubleTypeSwitch &&) = delete;
  DoubleTypeSwitch(const DoubleTypeSwitch &) = delete;
  DoubleTypeSwitch &operator=(DoubleTypeSwitch &&) = delete;
  DoubleTypeSwitch &operator=(const DoubleTypeSwitch &) = delete;

public:
  ~DoubleTypeSwitch() {
    u = nullptr;
    v = nullptr;
  }

  static auto Switch(const U &u, const V &v) { return DoubleTypeSwitch(u, v); }

  template <typename UU, typename VV, bool Symmetric = false,
            typename Callback = void>
  DoubleTypeSwitch &Case(Callback &&callback) {
    if (!result) {
      if (auto uu = llvm::dyn_cast<UU>(*const_cast<U *>(u)))
        if (auto vv = llvm::dyn_cast<VV>(*const_cast<V *>(v))) {
          if constexpr (Symmetric)
            result = callback(uu, vv, false);
          else
            result = callback(uu, vv);
        }
      if constexpr (Symmetric) {
        if (!result)
          if (auto uu = llvm::dyn_cast<VV>(*const_cast<U *>(u)))
            if (auto vv = llvm::dyn_cast<UU>(*const_cast<V *>(v)))
              result = callback(vv, uu, true);
      }
    }
    return *this;
  }

  template <typename UU, typename VV, bool Symmetric = false>
  DoubleTypeSwitch &CaseValue(Result &&value) {
    if (!result) {
      if (auto uu = llvm::dyn_cast<UU>(*const_cast<U *>(u)))
        if (auto vv = llvm::dyn_cast<VV>(*const_cast<V *>(v)))
          result = std::forward<Result>(value);
      if constexpr (Symmetric) {
        if (!result)
          if (auto uu = llvm::dyn_cast<VV>(*const_cast<U *>(u)))
            if (auto vv = llvm::dyn_cast<UU>(*const_cast<V *>(v)))
              result = std::forward<Result>(value);
      }
    }
    return *this;
  }

  template <typename Callback>
  DoubleTypeSwitch &Default(Callback &&callback) {
    if (!result)
      result = callback(*const_cast<U *>(u), *const_cast<V *>(v));
    return *this;
  }

  DoubleTypeSwitch &DefaultValue(Result &&value) {
    if (!result)
      result = std::forward<Result>(value);
    return *this;
  }

  DoubleTypeSwitch &Default() {
    if (!result)
      result = Result{};
    return *this;
  }

  operator Result() { return getResult(); }

  Result getResult() {
    assert(result.has_value());
    return result.value();
  }
};
} // namespace xblang

#endif
