#include "xblang/Dialect/XBLang/IR/Type.h"
#include "mlir/IR/Builders.h"
#include "xblang/Dialect/XBLang/IR/Dialect.h"

#include "llvm/ADT/Bitfields.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetVector.h"

#define GET_TYPEDEF_CLASSES
#include "xblang/Dialect/XBLang/IR/XBLangTypes.cpp.inc"

namespace xblang {
namespace xb {
namespace detail {
class NamedTypeStorage : public mlir::TypeStorage {
public:
  using KeyTy = StringRef;

  NamedTypeStorage(StringRef name) : name(name), containedType(nullptr) {}

  bool operator==(const KeyTy &key) const { return key == name; }

  static NamedTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                     const KeyTy &key) {
    return new (allocator.allocate<NamedTypeStorage>())
        NamedTypeStorage(allocator.copyInto(key));
  }

  LogicalResult mutate(mlir::TypeStorageAllocator &, Type body) {
    containedType = body;
    return success();
  }

  StringRef getIdentifier() const { return name; }

  Type getType() const { return containedType; }

private:
  StringRef name;
  Type containedType;
};
} // namespace detail

//===----------------------------------------------------------------------===//
// Struct type.
//===----------------------------------------------------------------------===//

NamedType NamedType::get(MLIRContext *context, StringRef name, Type type) {
  auto named = Base::get(context, name);
  if (type)
    (void)named.setType(type);
  return named;
}

Type NamedType::getType() const { return getImpl()->getType(); }

StringRef NamedType::getName() const { return getImpl()->getIdentifier(); }

LogicalResult NamedType::setType(Type type) { return Base::mutate(type); }

bool NamedType::isOpaque() const { return getType() == Type(); }

::mlir::Type NamedType::parse(::mlir::AsmParser &odsParser) {
  thread_local mlir::SetVector<StringRef> knownStructNames;
  unsigned stackSize = knownStructNames.size();
  (void)stackSize;
  auto guard = llvm::make_scope_exit([&]() {
    assert(knownStructNames.size() == stackSize &&
           "malformed identified stack when parsing recursive types");
  });
  Location loc = odsParser.getEncodedSourceLoc(odsParser.getCurrentLocation());
  if (failed(odsParser.parseLess()))
    return NamedType();
  std::string name;
  if (failed(odsParser.parseString(&name)))
    return {};
  if (knownStructNames.count(name)) {
    if (failed(odsParser.parseGreater()))
      return {};
    return NamedType::get(loc.getContext(), name);
  }
  if (failed(odsParser.parseComma()))
    return NamedType();
  if (succeeded(odsParser.parseOptionalColon())) {
    Type type;
    knownStructNames.insert(name);
    auto result = odsParser.parseType(type);
    knownStructNames.pop_back();
    if (failed(result))
      return {};
    return NamedType::get(loc.getContext(), name, type);
  }
  return NamedType::get(loc.getContext(), name);
}

void NamedType::print(::mlir::AsmPrinter &odsPrinter) const {
  thread_local mlir::SetVector<StringRef> knownStructNames;
  unsigned stackSize = knownStructNames.size();
  (void)stackSize;
  auto guard = llvm::make_scope_exit([&]() {
    assert(knownStructNames.size() == stackSize &&
           "malformed identified stack when printing recursive types");
  });

  odsPrinter << '<';
  odsPrinter << '"' << getName() << '"';
  if (knownStructNames.count(getName())) {
    odsPrinter << '>';
    return;
  }
  auto type = getType();
  if (type) {
    odsPrinter << ": ";
    knownStructNames.insert(getName());
    odsPrinter.printType(type);
    knownStructNames.pop_back();
  }
  odsPrinter << '>';
}

static ::mlir::OptionalParseResult
XblangTypeParser(::mlir::AsmParser &odsParser, ::llvm::StringRef *mnemonic,
                 ::mlir::Type &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(
             odsParser)
      .Case(::xblang::xb::NamedType::getMnemonic(),
            [&](llvm::StringRef, llvm::SMLoc) {
              value = ::xblang::xb::NamedType::parse(odsParser);
              return ::mlir::success(!!value);
            })
      .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
        *mnemonic = keyword;
        return std::nullopt;
      });
}

::mlir::Type
XBLangDialect::parseType(::mlir::DialectAsmParser &odsParser) const {
  ::llvm::SMLoc typeLoc = odsParser.getCurrentLocation();
  ::llvm::StringRef mnemonic;
  ::mlir::Type genType;
  auto parseResult = generatedTypeParser(odsParser, &mnemonic, genType);
  if (parseResult.has_value())
    return genType;

  parseResult = XblangTypeParser(odsParser, &mnemonic, genType);
  if (parseResult.has_value())
    return genType;

  odsParser.emitError(typeLoc) << "unknown  type `" << mnemonic
                               << "` in dialect `" << getNamespace() << "`";
  return {};
}

static ::mlir::LogicalResult XblangTypePrinter(::mlir::Type def,
                                               ::mlir::AsmPrinter &odsPrinter) {
  return ::llvm::TypeSwitch<::mlir::Type, ::mlir::LogicalResult>(def)
      .Case<::xblang::xb::NamedType>([&](auto t) {
        odsPrinter << ::xblang::xb::NamedType::getMnemonic();
        t.print(odsPrinter);
        return ::mlir::success();
      })
      .Default([](auto) { return ::mlir::failure(); });
}

void XBLangDialect::printType(::mlir::Type type,
                              ::mlir::DialectAsmPrinter &odsPrinter) const {
  if (::mlir::succeeded(generatedTypePrinter(type, odsPrinter)))
    return;
  if (::mlir::succeeded(XblangTypePrinter(type, odsPrinter)))
    return;
}

void XBLangDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "xblang/Dialect/XBLang/IR/XBLangTypes.cpp.inc"
      >();
  addTypes<::xblang::xb::NamedType>();
}
} // namespace xb
} // namespace xblang

namespace xblang {
std::pair<mlir::Type, int> arithmeticTypePromotion(mlir::Type t1,
                                                   mlir::Type t2) {
  if ((!t1 || !t2) || (!t1.isIntOrIndexOrFloat() || !t2.isIntOrIndexOrFloat()))
    return {nullptr, -1};
  else if (t1 == t2)
    return {t1, 0};
  else if (t1.isa<mlir::FloatType>() && t2.isa<mlir::FloatType>()) {
    auto w1 = t1.getIntOrFloatBitWidth();
    auto w2 = t2.getIntOrFloatBitWidth();
    if (w1 > w2)
      return {t1, 1};
    else
      return {t2, 2};
  } else if (t1.isa<mlir::FloatType>() && t2.isa<mlir::IntegerType>())
    return {t1, 1};
  else if (t1.isa<mlir::IntegerType>() && t2.isa<mlir::FloatType>())
    return {t2, 2};
  auto i1 = t1.dyn_cast<mlir::IntegerType>();
  auto i2 = t2.dyn_cast<mlir::IntegerType>();
  if (i1 && i2) {
    auto w1 = i1.getWidth();
    auto w2 = i2.getWidth();
    if (w1 > w2)
      return {t1, 1};
    else if (w1 < w2)
      return {t2, 2};
    else {
      if (i1.isUnsigned())
        return {t1, 1};
      else
        return {t2, 2};
    }
  }
  return {nullptr, -1};
}
} // namespace xblang
