#include "toy/ast/ast.hpp"

#include <llvm/ADT/TypeSwitch.h>

#include <ctl/object/numerics.hpp>
#include <fmt/ranges.h>

#include <ranges>

using namespace toy;

namespace {
/// RAII helper to manage increasing/decreasing the indentation as we traverse
/// the AST
struct Indent {
  explicit Indent(i32& level) : level(&level) { ++level; }

  Indent(Indent&& other) noexcept {
    level = std::exchange(other.level, nullptr);
  }

  ~Indent() {
    if (level) --*level;
  }

  i32* level;
};

/// Helper class that implement the AST tree traversal and print the nodes along
/// the way. The only data member is the current indentation level.
class ASTDumper {
 public:
  void dump(const ModuleAST& node);

 private:
  void dump(const VarType& type) const;
  void dump(const VarDeclExprAST& var_decl);
  void dump(const ExprAST& expr);
  void dump(std::span<const std::unique_ptr<ExprAST>> expr_list);
  void dump(const NumberExprAST& num);
  void dump(const LiteralExprAST& node);
  void dump(const VariableExprAST& node);
  void dump(const ReturnExprAST& node);
  void dump(const BinaryExprAST& node);
  void dump(const CallExprAST& node);
  void dump(const PrintExprAST& node);
  void dump(const PrototypeAST& node);
  void dump(const FunctionAST& node);

  /// Actually print spaces matching the current indentation level
  Indent indent() {
    Indent level(current_indent);
    reindent();
    return level;
  }

  void reindent() const {
    for (i32 i = 0; i < current_indent; ++i) fmt::print("  ");
  }

  i32 current_indent = 0;
};

/// Dispatch to a generic expressions to the appropriate subclass using RTTI
void ASTDumper::dump(const ExprAST& expr) {
  llvm::TypeSwitch<const ExprAST*>(&expr)
      .Case<
          BinaryExprAST,
          CallExprAST,
          LiteralExprAST,
          NumberExprAST,
          PrintExprAST,
          ReturnExprAST,
          VarDeclExprAST,
          VariableExprAST>([&](const auto* node) { dump(*node); })
      .Default([&](const auto*) {
        // No match, fallback to a generic message
        auto i_ = indent();
        fmt::println(
            "<unknown Expr, kind {}>", std::to_underlying(expr.getKind())
        );
      });
}

/// A variable declaration is printing the variable name, the type, and then
/// recurse in the initializer value.
void ASTDumper::dump(const VarDeclExprAST& var_decl) {
  auto i_ = indent();
  fmt::print("VarDecl {}", var_decl.name());
  dump(var_decl.type());
  fmt::println(" {}", var_decl.location());
  dump(var_decl.init_val());
}

/// A "block", or a list of expression
void ASTDumper::dump(std::span<const std::unique_ptr<ExprAST>> expr_list) {
  auto i_ = indent();
  fmt::println("Block {{");
  for (const auto& expr : expr_list) dump(*expr);
  reindent();
  fmt::println("}} // Block");
}

/// A literal number, just print the value.
void ASTDumper::dump(const NumberExprAST& num) {
  auto i_ = indent();
  fmt::println("{} {}", num.value(), num.location());
}

/// Helper to print recursively a literal. This handles nested array like:
///    [ [ 1, 2 ], [ 3, 4 ] ]
/// We print out such array with the dimensions spelled out at every level:
///    <2, 2>[<2>[ 1, 2 ], <2>[ 3, 4 ] ]
void print_lit_helper(const ExprAST& lit_or_num) {
  // Inside a literal expression we can have either a number or another literal
  if (auto* num = llvm::dyn_cast<NumberExprAST>(&lit_or_num)) {
    fmt::print("{}", num->value());
    return;
  }
  const auto& literal = llvm::cast<LiteralExprAST>(lit_or_num);

  // Print the dimension for this literal first
  fmt::print("<{}>", fmt::join(literal.dims(), ", "));

  // Now print the content, recursing on every element of the list
  fmt::print("[ ");
  for (bool is_first = true; const auto& val : literal.values()) {
    if (is_first) is_first = false;
    else fmt::print(", ");
    print_lit_helper(*val);
  }
  fmt::print(" ]");
}

/// Print a literal, see the recursive helper above for the implementation.
void ASTDumper::dump(const LiteralExprAST& node) {
  auto i_ = indent();
  fmt::print("Literal: ");
  print_lit_helper(node);
  fmt::println(" {}", node.location());
}

/// Print a variable reference (just a name).
void ASTDumper::dump(const VariableExprAST& node) {
  auto i_ = indent();
  fmt::println("var: {} {}", node.name(), node.location());
}

/// Return statement print the return and its (optional) argument.
void ASTDumper::dump(const ReturnExprAST& node) {
  auto i_ = indent();
  fmt::println("Return");
  if (node.expr()) dump(*node.expr());
  else {
    auto i_ = indent();
    fmt::println("(void)");
  }
}

/// Print a binary operation, first the operator, then recurse into LHS and RHS.
void ASTDumper::dump(const BinaryExprAST& node) {
  auto i_ = indent();
  fmt::println(
      "BinOp: {} {}", ctl::lossless_cast<char>(node.op()), node.location()
  );
  dump(node.lhs());
  dump(node.rhs());
}

/// Print a call expression, first the callee name and the list of args by
/// recursing into each individual argument.
void ASTDumper::dump(const CallExprAST& node) {
  auto i_ = indent();
  fmt::println("Call '{}' [ {}", node.callee(), node.location());
  for (const auto& arg : node.args()) dump(*arg);
  reindent();
  fmt::println("]");
}

/// Print a builtin print call, first the builtin name and then the argument.
void ASTDumper::dump(const PrintExprAST& node) {
  auto i_ = indent();
  fmt::println("Print [ {}", node.location());
  dump(node.arg());
  reindent();
  fmt::println("]");
}

/// Print type: only the shape is printed in between '<' and '>'
void ASTDumper::dump(const VarType& type) const {
  fmt::print("<{}>", fmt::join(type.shape, ", "));
}

/// Print a function prototype, first the function name, and then the list of
/// parameters names.
void ASTDumper::dump(const PrototypeAST& node) {
  auto i_ = indent();
  fmt::println("Proto '{}' {}", node.name(), node.location());
  reindent();
  fmt::println(
      "Params: [{}]",
      fmt::join(
          node.args() | std::views::transform([](const auto& arg) {
            return arg->name();
          }),
          ", "
      )
  );
}

/// Print a function, first the prototype and then the body.
void ASTDumper::dump(const FunctionAST& node) {
  auto i_ = indent();
  fmt::println("Function");
  dump(node.proto());
  dump(node.body());
}

/// Print a module, actually loop over the functions and print them in sequence.
void ASTDumper::dump(const ModuleAST& node) {
  fmt::println("Module:");
  for (const auto& f : node.functions()) dump(f);
}

} // namespace

void toy::dump_module_ast(const toy::ModuleAST& ast) { ASTDumper().dump(ast); }