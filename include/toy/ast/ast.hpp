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
      : kind(kind)
      , location(location) {}

  constexpr virtual ~ExprAST() = default;

  [[nodiscard]] constexpr ExprASTKind getKind() const { return kind; }

  [[nodiscard]] const Location& loc() const { return location; }

 private:
  const ExprASTKind kind;
  Location          location;
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
  constexpr LiteralExprAST(
      Location                              loc,
      std::vector<std::unique_ptr<ExprAST>> values,
      std::vector<i64>                      dims
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
  std::vector<i64>                      dims_;
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
      std::unique_ptr<ExprAST> initVal
  )
      : ExprAST(ExprASTKind::VarDecl, loc)
      , name_(std::move(name))
      , type_(std::move(type))
      , init_val_(std::move(initVal)) {}

  [[nodiscard]] constexpr std::string_view name() const { return name_; }

  [[nodiscard]] const ExprAST* init_val() const { return init_val_.get(); }

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
  ReturnExprAST(Location loc, std::optional<std::unique_ptr<ExprAST>> expr)
      : ExprAST(ExprASTKind::Return, loc)
      , expr_(std::move(expr)) {}

  [[nodiscard]] constexpr std::optional<const ExprAST*> expr() const {
    return expr_.has_value() ? std::optional(expr_->get()) : std::nullopt;
  }

  [[nodiscard]] static constexpr bool classof(const ExprAST* c) {
    return c->getKind() == ExprASTKind::Return;
  }

 private:
  std::optional<std::unique_ptr<ExprAST>> expr_;
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

  [[nodiscard]] const ExprAST* lhs() const { return lhs_.get(); }

  [[nodiscard]] const ExprAST* rhs() const { return rhs_.get(); }

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
  [[nodiscard]] constexpr CallExprAST(
      Location                              loc,
      std::string                           callee,
      std::vector<std::unique_ptr<ExprAST>> args
  )
      : ExprAST(ExprASTKind::Call, std::move(loc))
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
  PrintExprAST(Location loc, std::unique_ptr<ExprAST> arg)
      : ExprAST(ExprASTKind::Print, loc)
      , arg_(std::move(arg)) {}

  [[nodiscard]]  const ExprAST* arg() const { return arg_.get(); }

  [[nodiscard]] static constexpr bool classof(const ExprAST* c) {
    return c->getKind() == ExprASTKind::Print;
  }

 private:
  std::unique_ptr<ExprAST> arg_;
};

} // namespace toy

#endif // TOY_AST_AST_HPP