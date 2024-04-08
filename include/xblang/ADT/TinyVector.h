#ifndef XBLANG_ADT_TINYVECTOR_H
#define XBLANG_ADT_TINYVECTOR_H

#include "xblang/Support/CompareExtras.h"
#include "xblang/Support/STLExtras.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <numeric>

namespace xblang {
template <typename T>
class TinyVectorImpl;

template <typename T>
class alignas(8) TinyVectorImplResource {
private:
  using value_type = T;
  using pointer = value_type *;
  using size_type = uint32_t;
  template <typename>
  friend class xblang::TinyVectorImpl;

public:
  TinyVectorImplResource() = default;

  TinyVectorImplResource(size_type size) {
    if (size) {
      _capacity = size;
      _data = std::allocator<value_type>().allocate(_capacity);
    }
  }

  TinyVectorImplResource(pointer data, size_type size)
      : _data(data), _capacity(size) {}

  TinyVectorImplResource(TinyVectorImplResource &&other)
      : _data(std::exchange(other._data, nullptr)),
        _capacity(std::exchange(other._capacity, 0)) {}

  TinyVectorImplResource(const TinyVectorImplResource &&) = delete;

  ~TinyVectorImplResource() {
    if (_capacity)
      std::allocator<value_type>().deallocate(_data, _capacity);
    _data = nullptr;
    _capacity = 0;
  }

  TinyVectorImplResource &operator=(TinyVectorImplResource &&other) {
    _data = std::exchange(other._data, nullptr);
    _capacity = std::exchange(other._capacity, 0);
    return *this;
  }

  TinyVectorImplResource &operator=(const TinyVectorImplResource &&) = delete;

  pointer get() { return _data; }

  TinyVectorImplResource release() {
    return TinyVectorImplResource(std::exchange(_data, nullptr),
                                  std::exchange(_capacity, 0));
  }

  static void destroy_range(pointer begin, pointer end) {
    if (begin && begin < end)
      std::destroy(begin, end);
  }

  void destroy_range(size_type size) {
    if (_data && size)
      std::destroy(_data, _data + size);
  }

protected:
  pointer _data{};
  size_type _capacity{};
};

template <typename T>
class alignas(8) TinyVectorImpl : protected TinyVectorImplResource<T> {
private:
  using base = TinyVectorImplResource<T>;

public:
  using value_type = T;
  using size_type = uint32_t;
  using difference_type = std::ptrdiff_t;
  using pointer = value_type *;
  using const_pointer = const value_type *;
  using reference = value_type &;
  using const_reference = const value_type &;
  using iterator = pointer;
  using const_iterator = const_pointer;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  TinyVectorImpl() = default;

  TinyVectorImpl(size_type size, const value_type &value = {}) {
    resize(size, value);
  }

  TinyVectorImpl(TinyVectorImpl &&other) { move(other); }

  TinyVectorImpl(const TinyVectorImpl &other)
      : base(growthFunction(other._size)), _size(other._size) {
    std::uninitialized_copy(other.begin(), other.end(), begin());
  }

  TinyVectorImpl(const llvm::ArrayRef<T> &other)
      : base(growthFunction(other.size())), _size(other.size()) {
    std::uninitialized_copy(other.begin(), other.end(), begin());
  }

  ~TinyVectorImpl() {
    if (base::_capacity)
      base::destroy_range(_size);
    _size = 0;
  }

  TinyVectorImpl &operator=(TinyVectorImpl &&other) {
    move(other);
    return *this;
  }

  TinyVectorImpl &operator=(const TinyVectorImpl &other) {
    if (other._size > base::_capacity) {
      raw_reserve(other._size);
      std::uninitialized_copy(other.begin(), other.end(), begin());
    } else {
      std::copy(other.begin(), other.end(), begin());
      base::destroy_range(begin() + other._size, end());
    }
    _size = other._size;
    return *this;
  }

  TinyVectorImpl &operator=(const llvm::ArrayRef<T> &other) {
    if (other.size() > base::_capacity) {
      raw_reserve(other.size());
      std::uninitialized_copy(other.begin(), other.end(), begin());
    } else {
      std::copy(other.begin(), other.end(), begin());
      base::destroy_range(begin() + other.size(), end());
    }
    _size = other.size();
    return *this;
  }

  static constexpr size_type max_capacity() {
    return std::numeric_limits<size_type>::max();
  }

  void clear() {
    if (base::_capacity)
      base::destroy_range(begin(), end());
    _size = 0;
  }

  void reserve(size_type size) {
    if (size > base::_capacity) {
      auto old = base::release();
      static_cast<base &>(*this) = base(growthFunction(size));
      if (_size > 0)
        std::uninitialized_move(old.get(), old.get() + _size, base::_data);
      old.destroy_range(_size);
    }
  }

  void resize(size_type size, const value_type &value = {}) {
    reserve(size);
    if (size > _size)
      std::uninitialized_fill(end(), begin() + size, value);
    else
      base::destroy_range(begin() + size, end());
    _size = size;
  }

  void push_back(const value_type &value) {
    auto sz = _size + 1;
    reserve(sz);
    if (_size < base::_capacity) {
      construct_at(base::_data + _size, value);
      _size = sz;
    }
  }

  void push_back(value_type &&value) {
    auto sz = _size + 1;
    reserve(sz);
    if (_size < base::_capacity) {
      construct_at(base::_data + _size, std::move(value));
      _size = sz;
    }
  }

  void swap(TinyVectorImpl &other) {
    std::swap(other._data, base::_data);
    std::swap(other._capacity, base::_capacity);
    std::swap(other._size, _size);
  }

  void move(TinyVectorImpl &other) {
    base::_data = std::exchange(other._data, nullptr);
    base::_capacity = std::exchange(other._capacity, 0);
    _size = std::exchange(other._size, 0);
  }

protected:
  static inline size_type growthFunction(size_type size) {
    constexpr size_type max = max_capacity();
    if (size == 0 || (size & (size - 1)) == 0)
      return size;
    size_type l2 = llvm::Log2_32(size);
    return l2 >= 31 ? max : 1 << (l2 + 1);
  }

  iterator begin() { return this->_data; }

  iterator end() { return this->_data + _size; }

  const_iterator begin() const { return this->_data; }

  const_iterator end() const { return this->_data + _size; }

  void raw_reserve(size_type size) {
    if (size > base::_capacity) {
      auto old = base::release();
      static_cast<base &>(*this) = base(growthFunction(size));
      old.destroy_range(_size);
    }
  }

protected:
  size_type _size{};
};

template <typename T>
class alignas(8) TinyVector : public TinyVectorImpl<T> {
private:
  using base = TinyVectorImpl<T>;

public:
  using value_type = typename base::value_type;
  using size_type = typename base::size_type;
  using difference_type = typename base::difference_type;
  using pointer = typename base::pointer;
  using const_pointer = typename base::const_pointer;
  using reference = typename base::reference;
  using const_reference = typename base::const_reference;
  using iterator = typename base::iterator;
  using const_iterator = typename base::const_iterator;
  using reverse_iterator = typename base::reverse_iterator;
  using const_reverse_iterator = typename base::const_reverse_iterator;
  TinyVector() = default;
  ~TinyVector() = default;

  TinyVector(size_type size, const value_type &value = {})
      : base(size, value) {}

  TinyVector(TinyVector &&other)
      : base(static_cast<base &&>(std::move(other))) {}

  TinyVector(const TinyVector &other)
      : base(static_cast<const base &>(other)) {}

  TinyVector(const llvm::ArrayRef<T> &other) : base(other) {}

  TinyVector &operator=(TinyVector &&other) {
    base::operator=(static_cast<base &&>(std::move(other)));
    return *this;
  }

  TinyVector &operator=(const TinyVector &other) {
    base::operator=(static_cast<const base &>(other));
    return *this;
  }

  TinyVector &operator=(const llvm::ArrayRef<T> &other) {
    base::operator=(other);
    return *this;
  }

  pointer data() const { return this->_data; }

  reference operator[](size_t index) { return this->_data[index]; }

  const_reference operator[](size_t index) const { return this->_data[index]; }

  iterator begin() { return this->_data; }

  const_iterator begin() const { return this->_data; }

  iterator end() { return this->_data + this->_size; }

  const_iterator end() const { return this->_data + this->_size; }

  reverse_iterator rbegin() { return reverse_iterator(end()); }

  const_reverse_iterator rbegin() const {
    return const_reverse_iterator(end());
  }

  reverse_iterator rend() { return reverse_iterator(begin()); }

  const_reverse_iterator rend() const { return const_reverse_iterator(end()); }

  bool empty() const { return this->_size; }

  uint32_t size() const { return this->_size; }

  uint32_t capacity() const { return this->_capacity; }

  reference at(size_t index) {
    assert(index < this->_size);
    return this->_data[index];
  }

  const_reference at(size_t index) const {
    assert(index < this->_size);
    return this->_data[index];
  }

  reference front() {
    assert(this->_data);
    return *this->_data;
  }

  const_reference front() const {
    assert(this->_data);
    return *this->_data;
  }

  reference back() {
    assert(this->_data && this->_size);
    return *this->_data;
  }

  const_reference back() const {
    assert(this->_data && this->_size);
    return this->_data[this->_size - 1];
  }

  llvm::ArrayRef<value_type> array() const {
    return llvm::ArrayRef<value_type>(this->_data, this->_size);
  }

  auto compare(const TinyVector &other) const {
    return lexicographical_compare_three_way(begin(), end(), other.begin(),
                                             other.end());
  }

  auto compare(const llvm::ArrayRef<value_type> &other) const {
    return lexicographical_compare_three_way(begin(), end(), other.begin(),
                                             other.end());
  }

  auto operator<=>(const TinyVector &other) const { compare(other); }

  bool operator==(const TinyVector &other) const { return compare(other) == 0; }

  bool operator!=(const TinyVector &other) const { return compare(other) != 0; }

  bool operator<(const TinyVector &other) const { return compare(other) < 0; }

  bool operator<=(const TinyVector &other) const { return compare(other) <= 0; }

  bool operator>(const TinyVector &other) const { return compare(other) > 0; }

  bool operator>=(const TinyVector &other) const { return compare(other) >= 0; }

  auto operator<=>(const llvm::ArrayRef<value_type> &other) const {
    return compare(other);
  }

  bool operator==(const llvm::ArrayRef<value_type> &other) const {
    return compare(other) == 0;
  }

  bool operator!=(const llvm::ArrayRef<value_type> &other) const {
    return compare(other) != 0;
  }

  bool operator<(const llvm::ArrayRef<value_type> &other) const {
    return compare(other) < 0;
  }

  bool operator<=(const llvm::ArrayRef<value_type> &other) const {
    return compare(other) <= 0;
  }

  bool operator>(const llvm::ArrayRef<value_type> &other) const {
    return compare(other) > 0;
  }

  bool operator>=(const llvm::ArrayRef<value_type> &other) const {
    return compare(other) >= 0;
  }
};
} // namespace xblang

#endif
