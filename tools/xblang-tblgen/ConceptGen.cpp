//===- ConceptGen.cpp - Concept generator ------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Tablegen conceptDef generator.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/TableGen/GenInfo.h"
#include "xblang/Support/Format.h"
#include "xblang/TableGen/Concept.h"
#include "xblang/TableGen/Util.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

#include "TemplateEngine.h"

using namespace xblang;
using namespace xblang::tablegen;

namespace {
struct ConceptGen {
  using Environment = TextTemplate::Environment;
  using Op = Concept::Op;
  ConceptGen() = default;
  /// Emits the concepts.
  void emitDecls(const llvm::RecordKeeper &records, llvm::raw_ostream &os);
  /// Emit a concept.
  void emitDecl(Concept conceptDef, llvm::raw_ostream &os);
  /// Emits the concepts definition.
  void emitDefs(const llvm::RecordKeeper &records, llvm::raw_ostream &os);
  /// Emit the concept definition.
  void emitDef(Concept conceptDef, llvm::raw_ostream &os);
  /// Emits the pure concept interface.
  void emitPureInterfaceDecl(Concept conceptDef, Op &op, const Environment &env,
                             llvm::raw_ostream &os);
  /// Emits the model declaration.
  void emitModelDecl(Concept conceptDef, Op &op,
                     llvm::SmallVector<Concept> &ancestors,
                     const Environment &env, llvm::raw_ostream &os);
  /// Emits the concept interface.
  void emitConceptInterfaceDecl(Concept conceptDef, Op &op,
                                llvm::SmallVector<Concept> &ancestors,
                                const Environment &env, llvm::raw_ostream &os);
};
} // namespace

static llvm::cl::OptionCategory conceptGenCat("Options for -gen-concept-*");
static llvm::cl::opt<std::string>
    conceptClass("concept-class", llvm::cl::desc("The concept class to gen"),
                 llvm::cl::cat(conceptGenCat), llvm::cl::Optional,
                 llvm::cl::init(""));

void ConceptGen::emitDecls(const llvm::RecordKeeper &records,
                           llvm::raw_ostream &os) {
  os << "#ifdef GET_CLASS_DECL\n";
  for (llvm::Record *v : sortRecordDefinitions(
           records.getAllDerivedDefinitions(Concept::ClassType)))
    if (isInMainFile(v))
      if (auto conceptDef = Concept::castOrNull(v))
        emitDecl(*conceptDef, os);
  os << "#undef GET_CLASS_DECL\n";
  os << "#endif\n";
  os << "#ifdef GET_REGISTRATION_DECL\n";
  os << fmt("void register{0}Concepts(::xblang::XBContext&);\n", conceptClass);
  os << "#undef GET_REGISTRATION_DECL\n";
  os << "#endif\n";
}

void ConceptGen::emitDecl(Concept conceptDef, llvm::raw_ostream &os) {
  // Emit the name space.
  llvm::SmallVector<StringRef, 2> namespaces;
  llvm::SplitString(conceptDef.getCppNamespace(), namespaces, "::");
  for (StringRef ns : namespaces)
    os << "namespace " << ns << " {\n";
  // Define the base class.
  llvm::StringLiteral code = R"(
struct $className : public ::xblang::ConceptMixin<$className$parentConcepts> {
  friend class ::xblang::XBContext;
  using Base::Base;
  static constexpr std::string_view mnemonic = "$mnemonic";
  static constexpr std::string_view dialect_mnemonic = "$dialect_mnemonic";
${Interface}
$ConstructMethods
};
${ConceptInterface})";
  // Retrieve the parent concepts.
  std::string parentConcepts;
  for (auto c : conceptDef.getParentConcepts())
    parentConcepts +=
        (", " + c.getCppNamespace() + "::" + c.getClassName()).str();
  // Create the text template.
  auto tmpl = TemplateEngine::make(code);
  tmpl.insert("className", StrTemplate::make(conceptDef.getClassName()));
  tmpl.insert("parentConcepts", StrTemplate::make(parentConcepts));
  tmpl.insert("mnemonic", StrTemplate::make(conceptDef.getMnemonic()));
  tmpl.insert("dialect_mnemonic",
              StrTemplate::make(conceptDef.getDialectMnemonic()));
  // Create the interfaces.
  std::string interface;
  llvm::raw_string_ostream ios(interface);
  llvm::SmallVector<Concept> ancestors = getAncestorConcepts(conceptDef);
  Op op = conceptDef.getOp();
  emitPureInterfaceDecl(conceptDef, op, tmpl.getEnvironment(), ios);
  tmpl.insert("Interface", StrTemplate::make(interface));
  interface.clear();
  emitConceptInterfaceDecl(conceptDef, op, ancestors, tmpl.getEnvironment(),
                           ios);
  tmpl.insert("ConceptInterface", StrTemplate::make(interface));
  // Create the construct initialization methods.
  std::string construct;
  if (Construct::isa(&conceptDef.getDef()) || conceptDef.getPureConstruct()) {
    llvm::raw_string_ostream os(construct);
    os << R"(private:
  friend class ::xblang::XBContext;
  template <typename, typename...> friend class ::xblang::ConceptMixin;
  void initialize(::xblang::XBContext*);)";
    if (Construct::isa(&conceptDef.getDef()))
      emitModelDecl(conceptDef, op, ancestors, tmpl.getEnvironment(), os);
  }
  tmpl.insert("ConstructMethods", StrTemplate::make(construct));
  // Compile the template.
  os << tmpl.compile();
  for (StringRef ns : llvm::reverse(namespaces))
    os << "} // namespace " << ns << "\n";
  os << fmt("XB_DECLARE_TYPEINFO({0}::{1});\n", conceptDef.getCppNamespace(),
            conceptDef.getClassName());
}

void ConceptGen::emitDefs(const llvm::RecordKeeper &records,
                          llvm::raw_ostream &os) {
  os << "#ifdef GET_REGISTRATION_DEF\n";
  os << fmt("void register{0}Concepts(::xblang::XBContext& ctx){{\n",
            conceptClass);
  for (llvm::Record *v : sortRecordDefinitions(
           records.getAllDerivedDefinitions(Concept::ClassType)))
    if (isInMainFile(v))
      if (auto conceptDef = Concept::castOrNull(v))
        os << fmt("  ctx.registerConcept<{0}>();\n",
                  conceptDef->getClassName());
  os << "}\n";
  os << "#undef GET_REGISTRATION_DEF\n";
  os << "#endif\n";
  os << "#ifdef GET_CLASS_DEF\n";
  for (llvm::Record *v : sortRecordDefinitions(
           records.getAllDerivedDefinitions(Concept::ClassType)))
    if (isInMainFile(v))
      if (auto conceptDef = Concept::castOrNull(v))
        emitDef(*conceptDef, os);
  os << "#undef GET_CLASS_DEF\n";
  os << "#endif\n";
}

namespace {
void recursiveRegisterEmitter(llvm::raw_ostream &os,
                              const std::vector<Concept> &list) {
  for (Concept c : list) {
    os << fmt("  // BEGIN: {0}\n", c.getClassName());
    recursiveRegisterEmitter(os, c.getParentConcepts());
    os << fmt("  ctx->registerParentConcept<{0}::{1}>(id);\n",
              c.getCppNamespace(), c.getClassName());
    os << fmt("  // END: {0}\n", c.getClassName());
  }
}
} // namespace

void ConceptGen::emitDef(Concept conceptDef, llvm::raw_ostream &os) {
  // Emit the name space.
  llvm::SmallVector<StringRef, 2> namespaces;
  llvm::SplitString(conceptDef.getCppNamespace(), namespaces, "::");
  for (StringRef ns : namespaces)
    os << "namespace " << ns << " {\n";
  llvm::StringLiteral code = R"(
${className}Interface ${className}Interface::get(::xblang::XBContext *context,
                                                 ::mlir::Operation *op) {
  if (!op) return {};
  assert(context && "invalid context");
  return ${className}Interface(reinterpret_cast<Interface*>(
      context->getInterface(op->getName().getTypeID(),
      ::xblang::TypeInfo::get<ConceptType>())),
    op);
}
)";
  TemplateEngine tmpl = TemplateEngine::make(code);
  tmpl.insert("className", StrTemplate::make(conceptDef.getClassName()));
  os << tmpl.compile();
  if (Construct::isa(&conceptDef.getDef()) || conceptDef.getPureConstruct()) {
    os << fmt("void {0}::initialize(::xblang::XBContext* ctx) {{\n",
              conceptDef.getClassName());
    os << fmt("  auto id = getTypeInfo();\n");
    recursiveRegisterEmitter(os, conceptDef.getParentConcepts());
    os << "}\n";
  }

  for (StringRef ns : llvm::reverse(namespaces))
    os << "} // namespace " << ns << "\n";
  os << fmt("XB_DEFINE_TYPEINFO({0}::{1});\n", conceptDef.getCppNamespace(),
            conceptDef.getClassName());
}

//===----------------------------------------------------------------------===//
// Generate concept interfaces
//===----------------------------------------------------------------------===//

namespace {
enum class InterfaceKind { Concept, Model, Interface };

std::string emitInterfaceMethods(Concept::Op &op,
                                 const TextTemplate::Environment &env,
                                 InterfaceKind kind) {
  llvm::StringRef attrCode;
  llvm::StringRef operandCode;
  switch (kind) {
  case InterfaceKind::Concept:
    attrCode = R"(
  $AttrType (*get${AttrName}Attr)(::mlir::Operation* op);
  void (*set${AttrName}Attr)(::mlir::Operation* op, $AttrType attr);)";
    operandCode = R"(
  ${ValueType} (*get${OperandName})(::mlir::Operation* op);
  ${MutableValue} (*get${OperandName}Mutable)(::mlir::Operation* op);)";
    break;
  case InterfaceKind::Model:
    attrCode = R"(
  static $AttrType get${AttrName}AttrImpl(::mlir::Operation* op) {
    return ::mlir::cast<ConcreteOp>(op).get${AttrName}Attr();
  }
  static void set${AttrName}AttrImpl(::mlir::Operation* op, $AttrType attr) {
    ::mlir::cast<ConcreteOp>(op).set${AttrName}Attr(attr);
  })";
    operandCode = R"(
  static ${ValueType} get${OperandName}Impl(::mlir::Operation* op) {
    return ::mlir::cast<ConcreteOp>(op).get${OperandName}();
  }
  static ${MutableValue} get${OperandName}MutableImpl(::mlir::Operation* op) {
    return ::mlir::cast<ConcreteOp>(op).get${OperandName}Mutable();
  })";
    break;
  case InterfaceKind::Interface:
    attrCode = R"(
  inline $AttrType get${AttrName}Attr() const {
    return getImpl()->get${AttrName}Attr(getOp());
  }
  inline void set${AttrName}Attr($AttrType attr) const {
    getImpl()->set${AttrName}Attr(getOp(), attr);
  })";
    operandCode = R"(
  inline ${ValueType} get${OperandName}() const {
    return getImpl()->get${OperandName}(getOp());
  }
  inline ${MutableValue} get${OperandName}Mutable() const {
    return getImpl()->get${OperandName}Mutable(getOp());
  })";
    break;
  }
  std::string methods;
  llvm::raw_string_ostream os(methods);
  for (auto attr : op.attrs) {
    auto tmpl = TemplateEngine::make(attrCode);
    tmpl.insert("AttrName", StrTemplate::make(llvm::convertToCamelFromSnakeCase(
                                attr.name, true)));
    tmpl.insert("AttrType", StrTemplate::make(attr.attr.getStorageType()));
    os << tmpl.compile(env);
  }
  for (auto operand : op.operands) {
    auto tmpl = TemplateEngine::make(operandCode);
    tmpl.insert("OperandName",
                StrTemplate::make(
                    llvm::convertToCamelFromSnakeCase(operand.name, true)));
    auto ty = operand.constraint.getCPPClassName();
    if (ty == "::mlir::Type")
      ty = "::mlir::Value";
    else
      ty = fmt("::mlir::TypedValue<{0}>", ty);
    if (operand.isVariadic()) {
      tmpl.insert("ValueType",
                  StrTemplate::make("::mlir::Operation::operand_range"));
      tmpl.insert("MutableValue",
                  StrTemplate::make("::mlir::MutableOperandRange"));
    } else if (operand.isOptional()) {
      tmpl.insert("ValueType", StrTemplate::make(ty));
      tmpl.insert("MutableValue",
                  StrTemplate::make("::mlir::MutableOperandRange"));
    } else {
      tmpl.insert("ValueType", StrTemplate::make(ty));
      tmpl.insert("MutableValue", StrTemplate::make("::mlir::OpOperand&"));
    }
    os << tmpl.compile(env);
  }
  return methods;
}

std::string emitModelConstructor(Concept::Op &op,
                                 const TextTemplate::Environment &env) {
  llvm::StringRef attrCode = R"(
  this->get${AttrName}Attr = get${AttrName}AttrImpl;
  this->set${AttrName}Attr = set${AttrName}AttrImpl;)";
  llvm::StringRef operandCode = R"(
  this->get${OperandName} = get${OperandName}Impl;
  this->get${OperandName}Mutable = get${OperandName}MutableImpl;)";
  std::string str;
  llvm::raw_string_ostream os(str);
  for (auto attr : op.attrs) {
    auto tmpl = TemplateEngine::make(attrCode);
    tmpl.insert("AttrName", StrTemplate::make(llvm::convertToCamelFromSnakeCase(
                                attr.name, true)));
    os << tmpl.compile(env);
  }
  for (auto operand : op.operands) {
    auto tmpl = TemplateEngine::make(operandCode);
    tmpl.insert("OperandName",
                StrTemplate::make(
                    llvm::convertToCamelFromSnakeCase(operand.name, true)));
    os << tmpl.compile(env);
  }
  return str;
}

} // namespace

void ConceptGen::emitPureInterfaceDecl(Concept conceptDef, Op &op,
                                       const Environment &env,
                                       llvm::raw_ostream &os) {
  llvm::StringRef code = R"(
  /// Concept contract.
  struct PureInterface ${ParentConcepts} {
    using ConceptBase = ${className};${Methods}
  };)";
  auto tmpl = TemplateEngine::make(code);
  tmpl.insert("Methods", TemplateEngine::make(emitInterfaceMethods(
                             op, env, InterfaceKind::Concept)));
  std::string parentConcepts;
  llvm::raw_string_ostream pos(parentConcepts);
  llvm::interleaveComma(conceptDef.getParentConcepts(), pos, [&](Concept cep) {
    pos << fmt("public {0}::{1}::PureInterface", cep.getCppNamespace(),
               cep.getClassName());
  });
  if (!parentConcepts.empty())
    parentConcepts = ": " + parentConcepts;
  tmpl.insert("ParentConcepts", StrTemplate::make(parentConcepts));
  os << tmpl.compile(env);
}

void ConceptGen::emitConceptInterfaceDecl(Concept conceptDef, Op &op,
                                          llvm::SmallVector<Concept> &ancestors,
                                          const Environment &env,
                                          llvm::raw_ostream &os) {
  llvm::StringRef code = R"(
  /// Public concept interface returned by the XB context.
  struct ${className}Interface :
    public ::xblang::ConceptInterface<${className}> {
    using ::xblang::ConceptInterface<${className}>::ConceptInterface;
    static ${className}Interface get(::xblang::XBContext* context, ::mlir::Operation* op);
    ${Methods}
  };
)";
  auto tmpl = TemplateEngine::make(code);
  std::string methods =
      ::emitInterfaceMethods(op, env, InterfaceKind::Interface);
  for (Concept cep : ancestors) {
    auto op = cep.getOp();
    methods += ::emitInterfaceMethods(op, env, InterfaceKind::Interface);
  }
  tmpl.insert("Methods", TemplateEngine::make(methods));
  os << tmpl.compile(env);
}

void ConceptGen::emitModelDecl(Concept conceptDef, Op &op,
                               llvm::SmallVector<Concept> &ancestors,
                               const Environment &env, llvm::raw_ostream &os) {
  llvm::StringRef code = R"(
  /// Model used to build the constructs.
  template <typename ConcreteOp>
  struct Model : public ModelInterface, public PureInterface {${Methods}
  private:
    friend class ::xblang::XBContext;
    Model() {$Constructor
    }
  };)";
  auto tmpl = TemplateEngine::make(code);
  std::string methods = ::emitInterfaceMethods(op, env, InterfaceKind::Model);
  std::string constructor = ::emitModelConstructor(op, env);
  for (Concept cep : ancestors) {
    auto op = cep.getOp();
    methods += ::emitInterfaceMethods(op, env, InterfaceKind::Model);
    constructor += ::emitModelConstructor(op, env);
  }
  tmpl.insert("Methods", TemplateEngine::make(methods));
  tmpl.insert("Constructor", StrTemplate::make(constructor));
  os << tmpl.compile(env);
}

//===----------------------------------------------------------------------===//
// Generate the lexer MLIR module.
//===----------------------------------------------------------------------===//
static mlir::GenRegistration genConceptDecls(
    "gen-concept-decls", "Generate concept declarations",
    [](const llvm::RecordKeeper &records, llvm::raw_ostream &os) {
      ConceptGen().emitDecls(records, os);
      return false;
    });

static mlir::GenRegistration genConceptDefs(
    "gen-concept-defs", "Generate concept definitions",
    [](const llvm::RecordKeeper &records, llvm::raw_ostream &os) {
      ConceptGen().emitDefs(records, os);
      return false;
    });
