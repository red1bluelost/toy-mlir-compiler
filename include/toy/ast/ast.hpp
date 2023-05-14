#ifndef TOY_AST_AST_HPP
#define TOY_AST_AST_HPP

#include "toy/lexer/location.hpp"

#include <llvm/ADT/SmallVector.h>

#include <memory>
#include <optional>
#include <span>
#include <vector>

/// Base class for all expression nodes.
namespace toy {

/// A variable type with shape information.
struct VarType {
  llvm::SmallVector<i64, 2> shape;
};

class ExprAST {
 public:
  enum class ExprASTKind {
    VarDecl,
    Return,
    Num,
    Literal,
    Var,
    BinOp,
    Call,
    Print,
  };

  constexpr ExprAST(ExprASTKind kind, Location location)
      : kind_(kind)
      , location_(location) {}

  constexpr virtual ~ExprAST() = default;

  [[nodiscard]] constexpr ExprASTKind getKind() const { return kind_; }

  [[nodiscard]] const Location& location() const { return location_; }

 private:
  const ExprASTKind kind_;
  Location          location_;
};

/// Block-list of expressions.
using ExprASTList = std::vector<std::unique_ptr<ExprAST>>;

/// Expression class for numeric literals like "1.0".
class NumberExprAST : public ExprAST {
 public:
  constexpr NumberExprAST(Location loc, f64 value)
      : ExprAST(ExprASTKind::Num, loc)
      , value_(value) {}

  [[nodiscard]] constexpr f64 value() const { return value_; }

  [[nodiscard]] static constexpr bool classof(const ExprAST* c) {
    return c->getKind() == ExprASTKind::Num;
  }

 private:
  f64 value_;
};

/// Expression class for a literal value.
class LiteralExprAST : public ExprAST {
 public:
  LiteralExprAST(
      Location                              loc,
      std::vector<std::unique_ptr<ExprAST>> values,
      llvm::SmallVector<i64, 2>             dims
  )
      : ExprAST(ExprASTKind::Literal, loc)
      , values_(std::move(values))
      , dims_(std::move(dims)) {}

  [[nodiscard]] constexpr std::span<const std::unique_ptr<ExprAST>>
  values() const {
    return values_;
  }

  [[nodiscard]] constexpr std::span<const i64> dims() const { return dims_; }

  [[nodiscard]] static constexpr bool classof(const ExprAST* c) {
    return c->getKind() == ExprASTKind::Literal;
  }

 private:
  std::vector<std::unique_ptr<ExprAST>> values_;
  llvm::SmallVector<i64, 2>             dims_;
};

/// Expression class for referencing a variable, like "a".
class VariableExprAST : public ExprAST {
 public:
  [[nodiscard]] constexpr VariableExprAST(Location loc, std::string name)
      : ExprAST(ExprASTKind::Var, loc)
      , name_(std::move(name)) {}

  [[nodiscard]] constexpr std::string_view name() const { return name_; }

  [[nodiscard]] static constexpr bool classof(const ExprAST* c) {
    return c->getKind() == ExprASTKind::Var;
  }

 private:
  std::string name_;
};

/// Expression class for defining a variable.
class VarDeclExprAST : public ExprAST {
 public:
  VarDeclExprAST(
      Location                 loc,
      std::string              name,
      VarType                  type,
      std::unique_ptr<ExprAST> init_val
  )
      : ExprAST(ExprASTKind::VarDecl, loc)
      , name_(std::move(name))
      , type_(std::move(type))
      , init_val_(std::move(init_val)) {}

  [[nodiscard]] constexpr std::string_view name() const { return name_; }

  [[nodiscard]] constexpr const ExprAST& init_val() const { return *init_val_; }

  [[nodiscard]] constexpr const VarType& type() const { return type_; }

  [[nodiscard]] static constexpr bool classof(const ExprAST* c) {
    return c->getKind() == ExprASTKind::VarDecl;
  }

 private:
  std::string              name_;
  VarType                  type_;
  std::unique_ptr<ExprAST> init_val_;
};

/// Expression class for a return operator.
class ReturnExprAST : public ExprAST {
 public:
  explicit ReturnExprAST(Location loc, std::unique_ptr<ExprAST> expr = nullptr)
      : ExprAST(ExprASTKind::Return, loc)
      , expr_(std::move(expr)) {}

  [[nodiscard]] constexpr const ExprAST* expr() const { return expr_.get(); }

  [[nodiscard]] static constexpr bool classof(const ExprAST* c) {
    return c->getKind() == ExprASTKind::Return;
  }

 private:
  std::unique_ptr<ExprAST> expr_;
};

/// Expression class for a binary operator.
class BinaryExprAST : public ExprAST {
 public:
  BinaryExprAST(
      Location                 loc,
      u8                       op, // TODO: pick op type
      std::unique_ptr<ExprAST> lhs,
      std::unique_ptr<ExprAST> rhs
  )
      : ExprAST(ExprASTKind::BinOp, loc)
      , op_(op)
      , lhs_(std::move(lhs))
      , rhs_(std::move(rhs)) {}

  [[nodiscard]] constexpr u8 op() const { return op_; }

  [[nodiscard]] constexpr const ExprAST& lhs() const { return *lhs_; }

  [[nodiscard]] constexpr const ExprAST& rhs() const { return *rhs_; }

  [[nodiscard]] static constexpr bool classof(const ExprAST* c) {
    return c->getKind() == ExprASTKind::BinOp;
  }

 private:
  u8                       op_; // TODO: pick op type
  std::unique_ptr<ExprAST> lhs_, rhs_;
};

/// Expression class for function calls.
class CallExprAST : public ExprAST {
 public:
  constexpr CallExprAST(
      Location                              loc,
      std::string                           callee,
      std::vector<std::unique_ptr<ExprAST>> args
  )
      : ExprAST(ExprASTKind::Call, loc)
      , callee_(std::move(callee))
      , args_(std::move(args)) {}

  [[nodiscard]] constexpr std::string_view callee() const { return callee_; }

  [[nodiscard]] constexpr std::span<const std::unique_ptr<ExprAST>>
  args() const {
    return args_;
  }

  [[nodiscard]] static constexpr bool classof(const ExprAST* c) {
    return c->getKind() == ExprASTKind::Call;
  }

 private:
  std::string                           callee_;
  std::vector<std::unique_ptr<ExprAST>> args_;
};

/// Expression class for builtin print calls.
class PrintExprAST : public ExprAST {
 public:
  explicit PrintExprAST(Location loc, std::unique_ptr<ExprAST> arg)
      : ExprAST(ExprASTKind::Print, loc)
      , arg_(std::move(arg)) {}

  [[nodiscard]] const ExprAST& arg() const { return *arg_; }

  [[nodiscard]] static constexpr bool classof(const ExprAST* c) {
    return c->getKind() == ExprASTKind::Print;
  }

 private:
  std::unique_ptr<ExprAST> arg_;
};

/// This class represents the "prototype" for a function, which captures its
/// name, and its argument names (thus implicitly the number of arguments the
/// function takes).
class PrototypeAST {
 public:
  explicit PrototypeAST(
      Location                                      location,
      std::string                                   name,
      std::vector<std::unique_ptr<VariableExprAST>> args
  )
      : location_(location)
      , name_(std::move(name))
      , args_(std::move(args)) {}

  [[nodiscard]] const Location& location() const { return location_; }

  [[nodiscard]] std::string_view name() const { return name_; }

  [[nodiscard]] std::span<const std::unique_ptr<VariableExprAST>> args() const {
    return args_;
  }

 private:
  Location                                      location_;
  std::string                                   name_;
  std::vector<std::unique_ptr<VariableExprAST>> args_;
};

/// This class represents a function definition itself.
class FunctionAST {
 public:
  explicit FunctionAST(std::unique_ptr<PrototypeAST> proto, ExprASTList body)
      : proto_(std::move(proto))
      , body_(std::move(body)) {}

  [[nodiscard]] const PrototypeAST& proto() const { return *proto_; }

  [[nodiscard]] std::span<const std::unique_ptr<ExprAST>> body() const {
    return body_;
  }

 private:
  std::unique_ptr<PrototypeAST> proto_;
  ExprASTList                   body_;
};

/// This class represents a list of functions to be processed together
class ModuleAST {
 public:
  using FuncVector = std::vector<FunctionAST>;

  explicit ModuleAST(std::vector<FunctionAST> functions)
      : functions_(std::move(functions)) {}

  [[nodiscard]] std::span<const FunctionAST> functions() const {
    return functions_;
  }

 private:
  std::vector<FunctionAST> functions_;
};

void dump_module_ast(ModuleAST& ast);

} // namespace toy

#endif // TOY_AST_AST_HPP