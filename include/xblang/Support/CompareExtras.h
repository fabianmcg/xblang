#ifndef XBLANG_SUPPORT_COMPAREEXTRAS_H
#define XBLANG_SUPPORT_COMPAREEXTRAS_H

namespace xblang {
template <typename T, typename... Args>
bool isAnyOf(const T &target, const Args &...args) {
  return ((target == args) || ...);
}

template <typename T, typename... Args>
bool isNoneOf(const T &target, const Args &...args) {
  return ((target != args) && ...);
}

template <class I1Ty, class I2Ty, class CmpTy>
constexpr auto lexicographical_compare_three_way(I1Ty begin1, I1Ty end1,
                                                 I2Ty begin2, I2Ty end2,
                                                 CmpTy comp)
    -> decltype(comp(*begin1, *begin2)) {

  bool finished1 = (begin1 == end1);
  bool finished2 = (begin2 == end2);
  for (; !finished1 && !finished2;
       finished1 = (++begin1 == end1), finished2 = (++begin2 == end2))
    if (auto c = comp(*begin1, *begin2); c != 0)
      return c;

  return !finished1 ? 1 : !finished2 ? -1 : 0;
}

template <class I1Ty, class I2Ty>
constexpr int lexicographical_compare_three_way(I1Ty begin1, I1Ty end1,
                                                I2Ty begin2, I2Ty end2) {
  return lexicographical_compare_three_way(
      begin1, end1, begin2, end2, [](const auto &v1, const auto &v2) -> int {
        return v1 == v2 ? 0 : (v1 < v2 ? -1 : 1);
      });
}
} // namespace xblang

#endif
