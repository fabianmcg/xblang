#include "xblang/Sema/TypeSystem.h"
#include "xblang/ADT/DoubleTypeSwitch.h"

namespace xblang {
namespace xb {
XBLangTypeSystem::XBLangTypeSystem(MLIRContext &context) : context(&context) {}

XBLangTypeSystem::CastValidityKind
XBLangTypeSystem::castKindImpl(Type target, Type source) const {
  using Switch = ::xblang::DoubleTypeSwitch<Type, Type, CastValidityKind>;
  if (target == source)
    return Roundtrip;
  return Switch::Switch(target, source)
      .CaseValue<IntegerType, IntegerType>(Roundtrip)
      .CaseValue<FloatType, FloatType>(Roundtrip)
      .CaseValue<IntegerType, FloatType, true>(Roundtrip)
      .CaseValue<IntegerType, IndexType, true>(Roundtrip)
      .CaseValue<PointerType, AddressType, true>(Roundtrip)
      .CaseValue<PointerType, IndexType, true>(Roundtrip)
      .CaseValue<AddressType, IndexType, true>(Roundtrip)
      .DefaultValue(InvalidCast);
}

std::pair<XBLangTypeSystem::RankValidity, Type>
XBLangTypeSystem::rankTypesImpl(Type lhs, Type rhs) const {
  using Rank = std::pair<XBLangTypeSystem::RankValidity, Type>;
  using Switch = ::xblang::DoubleTypeSwitch<Type, Type, Rank>;
  if (lhs == rhs)
    return {RankValidity::Valid, lhs};
  return Switch::Switch(lhs, rhs)
      .Case<IntegerType, IntegerType>(
          [](IntegerType lhs, IntegerType rhs) -> Rank {
            auto w1 = lhs.getWidth();
            auto w2 = rhs.getWidth();
            if (w1 > w2)
              return {RankValidity::Valid, lhs};
            else if (w1 < w2)
              return {RankValidity::Valid, rhs};
            else {
              if (lhs.isUnsigned())
                return {RankValidity::Valid, lhs};
              else
                return {RankValidity::Valid, rhs};
            }
          })
      .Case<FloatType, FloatType>([](FloatType lhs, FloatType rhs) -> Rank {
        if (lhs.getWidth() > rhs.getWidth())
          return {RankValidity::Valid, lhs};
        else
          return {RankValidity::Valid, rhs};
      })
      .Case<IntegerType, FloatType, true>(
          [](IntegerType lhs, FloatType rhs, bool reverse) -> Rank {
            return {RankValidity::Valid, rhs};
          })
      .Case<IntegerType, IndexType, true>(
          [](IntegerType lhs, IndexType rhs, bool reverse) -> Rank {
            return {RankValidity::Valid, rhs};
          })
      .Case<PointerType, AddressType, true>(
          [](PointerType lhs, AddressType rhs, bool reverse) -> Rank {
            return {RankValidity::Valid, rhs};
          })
      .Case<PointerType, IndexType, true>(
          [](PointerType lhs, IndexType rhs, bool reverse) -> Rank {
            return {RankValidity::Valid, rhs};
          })
      .Case<PointerType, IntegerType, true>(
          [this](PointerType lhs, IntegerType rhs, bool reverse) -> Rank {
            return {RankValidity::Valid, Index()};
          })
      .Case<AddressType, IntegerType, true>(
          [this](AddressType lhs, IntegerType rhs, bool reverse) -> Rank {
            return {RankValidity::Valid, Index()};
          })
      .Case<AddressType, IndexType, true>(
          [](AddressType lhs, IndexType rhs, bool reverse) -> Rank {
            return {RankValidity::Valid, rhs};
          })
      .DefaultValue(Rank{RankValidity::Invalid, nullptr});
}

XBLangTypeSystem::CastValidityKind
XBLangTypeSystem::castSequenceImpl(Type target, Type source,
                                   SmallVector<Type, 2> &sequence) const {
  if (target == source)
    return SourceToTarget;
  CastValidityKind validity = InvalidCast;
  if (!isRef(target) && isRef(source)) {
    auto type = removeReference(source);
    sequence.push_back(type);
    validity = castSequenceImpl(target, type, sequence);
  } else {
    if (target.isIntOrFloat() && source.isIntOrFloat()) {
      sequence.push_back(target);
      validity = SourceToTarget;
    } else if (target.isIntOrIndex() && source.isIntOrIndex()) {
      sequence.push_back(target);
      validity = SourceToTarget;
    } else if (isAddressLike(target) && isAddressLike(source)) {
      if (!isIndex(source) && !isIndex(target))
        sequence.push_back(Index());
      sequence.push_back(target);
      validity = SourceToTarget;
    } else if (isAddressLike(target) && isa<IntegerType>(source)) {
      sequence.push_back(Index());
      if (!target.isIndex())
        sequence.push_back(target);
      validity = SourceToTarget;
    } else if (isAddressLike(target) && isa<FloatType>(source)) {
      sequence.push_back(Index());
      if (!target.isIndex())
        sequence.push_back(target);
      validity = SourceToTarget;
    } else if (isValidCast(target, source) != InvalidCast) {
      sequence.push_back(target);
      validity = SourceToTarget;
    }
  }
  if (validity == InvalidCast)
    sequence.clear();
  return validity;
}
} // namespace xb
} // namespace xblang
