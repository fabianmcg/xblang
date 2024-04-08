#ifndef XBLANG_ADT_IDENTIFIER_H
#define XBLANG_ADT_IDENTIFIER_H

#include "xblang/ADT/TinyVector.h"
#include "xblang/Support/CompareExtras.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
class raw_ostream;
}

namespace xblang {
class alignas(8) Identifier {
public:
  using size_type = uint32_t;
  using difference_type = std::ptrdiff_t;
  using pointer = const char *;
  using iterator = pointer;
  using reverse_iterator = std::reverse_iterator<iterator>;
  Identifier() = default;

  Identifier(Identifier &&identifier)
      : _data(std::exchange(identifier._data, nullptr)),
        extra(std::exchange(identifier.extra, 0)),
        _size(std::exchange(identifier._size, 0)) {}

  Identifier(const Identifier &identifier) {
    assert(identifier.isValid());
    if (identifier.isValid()) {
      _data = identifier._data;
      _size = identifier._size;
    }
  }

  Identifier(const char *_data, size_type _size) : _data(_data), _size(_size) {}

  Identifier(const llvm::StringRef &identifier)
      : _data(identifier.data()), _size(identifier.size()) {}

  ~Identifier() {
    if (extra == 0) {
      _data = nullptr;
      _size = 0;
    }
  }

  operator llvm::StringRef() const {
    assert(isValid());
    return toString();
  }

  operator bool() const { return _data && isValid(); }

  Identifier &operator=(Identifier &&identifier) {
    swap(identifier);
    return *this;
  }

  Identifier &operator=(const Identifier &identifier) {
    assert(isValid() && identifier.isValid());
    if (identifier.isValid()) {
      _data = identifier._data;
      _size = identifier._size;
    }
    return *this;
  }

  Identifier &operator=(const llvm::StringRef &other) {
    assert(isValid());
    _data = other.data();
    _size = other.size();
    return *this;
  }

  char operator[](size_t i) const {
    assert(isValid() && static_cast<size_t>(_size) > i);
    return _data[i];
  }

  iterator begin() const {
    assert(isValid());
    return _data;
  }

  iterator end() const {
    assert(isValid());
    return _data + _size;
  }

  bool isValid() const { return extra == 0; }

  auto compare(const Identifier &identifier) const {
    assert(isValid());
    return lexicographical_compare_three_way(begin(), end(), identifier.begin(),
                                             identifier.end());
  }

  auto compare(const llvm::StringRef &identifier) const {
    assert(isValid());
    return lexicographical_compare_three_way(begin(), end(), identifier.begin(),
                                             identifier.end());
  }

  bool operator==(const Identifier &identifier) const {
    return compare(identifier) == 0;
  }

  bool operator!=(const Identifier &identifier) const {
    return compare(identifier) != 0;
  }

  bool operator<(const Identifier &identifier) const {
    return compare(identifier) < 0;
  }

  bool operator<=(const Identifier &identifier) const {
    return compare(identifier) <= 0;
  }

  bool operator>(const Identifier &identifier) const {
    return compare(identifier) > 0;
  }

  bool operator>=(const Identifier &identifier) const {
    return compare(identifier) >= 0;
  }

  bool operator==(const llvm::StringRef &identifier) const {
    return compare(identifier) == 0;
  }

  bool operator!=(const llvm::StringRef &identifier) const {
    return compare(identifier) != 0;
  }

  bool operator<(const llvm::StringRef &identifier) const {
    return compare(identifier) < 0;
  }

  bool operator<=(const llvm::StringRef &identifier) const {
    return compare(identifier) <= 0;
  }

  bool operator>(const llvm::StringRef &identifier) const {
    return compare(identifier) > 0;
  }

  bool operator>=(const llvm::StringRef &identifier) const {
    return compare(identifier) >= 0;
  }

  llvm::StringRef toString() const {
    assert(isValid());
    return llvm::StringRef(_data, _size);
  }

  size_t size() const {
    assert(isValid());
    return _size;
  }

  pointer data() const {
    assert(isValid());
    return _data;
  }

  void swap(Identifier &other) {
    assert(isValid());
    _data = std::exchange(other._data, nullptr);
    extra = std::exchange(other.extra, 0);
    _size = std::exchange(other._size, 0);
  }

private:
  pointer _data{};
  size_type extra{};
  size_type _size{};
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &ost,
                              const Identifier &identifier);

class QualifiedIdentifier : protected TinyVector<Identifier> {
private:
  using base = TinyVector<Identifier>;

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
  QualifiedIdentifier() = default;

  QualifiedIdentifier(QualifiedIdentifier &&identifier)
      : base(static_cast<base &&>(std::move(identifier))) {}

  QualifiedIdentifier(const QualifiedIdentifier &identifier) {
    if (identifier.isQualified())
      base::operator=(static_cast<const base &>(identifier));
    else {
      _capacity = 0;
      _data = identifier._data;
      _size = identifier._size;
    }
  }

  QualifiedIdentifier(const Identifier &identifier) {
    setAsIdentifier(identifier.data(), identifier.size());
  }

  QualifiedIdentifier(const llvm::StringRef &identifier) {
    setAsIdentifier(identifier.data(), identifier.size());
  }

  QualifiedIdentifier(const llvm::ArrayRef<Identifier> &identifiers)
      : base(identifiers) {}

  QualifiedIdentifier(const llvm::ArrayRef<llvm::StringRef> &identifiers) {
    for (auto &identifier : identifiers)
      push_back(identifier);
  }

  ~QualifiedIdentifier() {
    if (_capacity == 0) {
      _data = nullptr;
      _size = 0;
    }
  }

  QualifiedIdentifier &operator=(QualifiedIdentifier &&identifier) {
    this->swap(identifier);
    return *this;
  }

  QualifiedIdentifier &operator=(const QualifiedIdentifier &identifier) {
    auto lhsQualified = isQualified();
    auto rhsQualified = identifier.isQualified();
    if (lhsQualified && rhsQualified)
      base::operator=(identifier);
    else if (lhsQualified && !rhsQualified) {
      resize(identifier.size());
      for (auto id : llvm::enumerate(identifier))
        (*this)[id.index()] = id.value();
    } else if (!lhsQualified && rhsQualified) {
      resetIdentifier();
      base::operator=(identifier);
    } else {
      _capacity = 0;
      _data = identifier._data;
      _size = identifier._size;
    }
    return *this;
  }

  operator llvm::StringRef() const {
    if (!size())
      return {};
    return operator[](0);
  }

  operator Identifier() const {
    if (!size())
      return {};
    return operator[](0);
  }

  bool isQualified() const { return _capacity > 0; }

  Identifier operator[](size_t i) const {
    if (isQualified())
      return base::operator[](i);
    assert(i == 0);
    return getAsIdentifier();
  }

  reference at(size_t i) {
    if (isQualified())
      return base::operator[](i);
    assert(i == 0);
    return getAsIdentifier();
  }

  const_reference at(size_t i) const {
    if (isQualified())
      return base::operator[](i);
    assert(i == 0);
    return getAsIdentifier();
  }

  iterator begin() {
    return isQualified() ? base::begin() : reinterpret_cast<Identifier *>(this);
  }

  iterator end() {
    return isQualified() ? base::end()
                         : (reinterpret_cast<Identifier *>(this) + 1);
  }

  const_iterator begin() const {
    return isQualified() ? base::begin()
                         : reinterpret_cast<Identifier const *>(this);
  }

  const_iterator end() const {
    return isQualified() ? base::end()
                         : (reinterpret_cast<Identifier const *>(this) + 1);
  }

  reverse_iterator rbegin() {
    return isQualified()
               ? base::rbegin()
               : reverse_iterator(reinterpret_cast<Identifier *>(this));
  }

  reverse_iterator rend() {
    return isQualified()
               ? base::rend()
               : reverse_iterator(reinterpret_cast<Identifier *>(this) + 1);
  }

  const_reverse_iterator rbegin() const {
    return isQualified() ? base::rbegin()
                         : const_reverse_iterator(
                               reinterpret_cast<Identifier const *>(this));
  }

  const_reverse_iterator rend() const {
    return isQualified() ? base::rend()
                         : const_reverse_iterator(
                               reinterpret_cast<Identifier const *>(this) + 1);
  }

  void push(const llvm::StringRef &str) {
    if (isQualified())
      push_back(str);
    else {
      if (_data) {
        auto data = std::exchange(_data, nullptr);
        auto size = std::exchange(_size, 0);
        reserve(2);
        push_back(Identifier(reinterpret_cast<const char *>(data), size));
        push_back(str);
      } else {
        _capacity = 0;
        _data = reinterpret_cast<Identifier *>(const_cast<char *>(str.data()));
        _size = str.size();
      }
    }
  }

  auto compare(const QualifiedIdentifier &other) const {
    return lexicographical_compare_three_way(begin(), end(), other.begin(),
                                             other.end());
  }

  bool operator==(const QualifiedIdentifier &identifier) const {
    return compare(identifier) == 0;
  }

  bool operator!=(const QualifiedIdentifier &identifier) const {
    return compare(identifier) != 0;
  }

  bool operator<(const QualifiedIdentifier &identifier) const {
    return compare(identifier) < 0;
  }

  bool operator<=(const QualifiedIdentifier &identifier) const {
    return compare(identifier) <= 0;
  }

  bool operator>(const QualifiedIdentifier &identifier) const {
    return compare(identifier) > 0;
  }

  bool operator>=(const QualifiedIdentifier &identifier) const {
    return compare(identifier) >= 0;
  }

  size_type size() const {
    return isQualified() ? base::size() : (_data ? 1 : 0);
  }

  using base::capacity;

  Identifier getBase() const {
    if (!size())
      return {};
    return operator[](0);
  }

private:
  void setAsIdentifier(const char *data, size_type size) {
    assert(!_capacity);
    if (_capacity == 0) {
      _data = reinterpret_cast<Identifier *>(const_cast<char *>(data));
      _size = size;
    }
  }

  void resetIdentifier() {
    assert(!_capacity);
    if (_capacity == 0) {
      _data = nullptr;
      _size = 0;
    }
  }

  Identifier &getAsIdentifier() {
    assert(!_capacity);
    return *reinterpret_cast<Identifier *>(this);
  }

  const Identifier &getAsIdentifier() const {
    assert(!_capacity);
    return *reinterpret_cast<Identifier const *>(this);
  }
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &ost,
                              const QualifiedIdentifier &identifier);
} // namespace xblang

#endif
