#include "toy/parser/parser.hpp"

#include "toy/ast/ast.hpp"

#include <llvm/Support/Casting.h>

#include <ctl/object/numerics.hpp>

#include <algorithm>
#include <cctype>
#include <optional>
#include <ranges>

using namespace toy;

std::unique_ptr<ReturnExprAST> Parser::parse_return() {
  Location loc = lexer.last_location();
  lexer.consume(Token::Return);

  // return takes an optional argument
  if (lexer.current_token() == Token::Semicolon)
    return std::make_unique<ReturnExprAST>(loc);

  auto expr = parse_expression();
  if (!expr) return nullptr;
  return std::make_unique<ReturnExprAST>(loc, std::move(expr));
}

std::unique_ptr<ExprAST> Parser::parse_number_expr() {
  auto result =
      std::make_unique<NumberExprAST>(lexer.last_location(), lexer.value());
  lexer.consume(Token::Number);
  return result;
}

std::unique_ptr<ExprAST> Parser::parse_tensor_literal_expr() {
  Location loc = lexer.last_location();
  lexer.consume(Token::SqrBracketOpen);

  // Hold the list of values at this nesting level.
  std::vector<std::unique_ptr<ExprAST>> values;
  // Hold the dimensions for all the nesting inside this level.
  llvm::SmallVector<i64, 2> dims;
  // Tracks whether a tensor literal was parsed
  bool contains_literals = false;
  while (true) {
    // We can have either another nested array or a number literal.
    if (lexer.current_token() == Token::SqrBracketOpen) {
      contains_literals = true;
      if (auto tensor = parse_tensor_literal_expr())
        values.push_back(std::move(tensor));
      else return nullptr; // parse error in the nested array.
    } else {
      if (lexer.current_token() != Token::Number)
        return parse_error<ExprAST>("<num> or [", "in literal expression");
      if (auto number = parse_number_expr())
        values.push_back(std::move(number));
      else return nullptr; // parse error in the nested array.
    }

    // End of this list on ']'
    if (lexer.current_token() == Token::SqrBracketClose) break;

    // Elements are separated by a comma.
    if (lexer.current_token() != Token::Comma)
      return parse_error<ExprAST>("] or ,", "in literal expression");

    lexer.consume(Token::Comma);
  }
  if (values.empty())
    return parse_error<ExprAST>("<something>", "to fill literal expression");
  lexer.consume(Token::SqrBracketClose);

  // Fill in the dimensions now. First the current nesting level:
  dims.push_back(ctl::lossless_cast<i64>(values.size()));

  // If there is any nested array, process all of them and ensure that
  // dimensions are uniform.
  if (contains_literals) {
    auto* first_literal = llvm::dyn_cast<LiteralExprAST>(values.front().get());
    if (!first_literal)
      return parse_error<ExprAST>(
          "uniform well-nested dimensions", "inside literal expression"
      );

    // Append the nested dimensions to the current level
    std::span<const i64> first_dims = first_literal->dims();
    dims.append(first_dims.begin(), first_dims.end());

    // Sanity check that shape is uniform across all elements of the list.
    if (std::ranges::any_of(
            values,
            [&](const LiteralExprAST* expr) {
              return !expr || !std::ranges::equal(expr->dims(), first_dims);
            },
            [](auto& expr) {
              return llvm::dyn_cast<LiteralExprAST>(expr.get());
            }
        ))
      return parse_error<ExprAST>(
          "uniform well-nested dimensions", "inside literal expression"
      );
  }

  return std::make_unique<LiteralExprAST>(
      loc, std::move(values), std::move(dims)
  );
}

std::unique_ptr<ExprAST> Parser::parse_paren_expr() {
  lexer.consume(Token::ParenOpen);
  auto expr = parse_expression();
  if (!expr) return nullptr;

  if (lexer.current_token() != Token::ParenClose)
    return parse_error<ExprAST>(")", "to close expression with parentheses");
  lexer.consume(Token::ParenClose);
  return expr;
}

std::unique_ptr<ExprAST> Parser::parse_identifier_expr() {
  std::string name = lexer.take_identifier();
  Location    loc  = lexer.last_location();
  lexer.consume(Token::Identifier);

  if (lexer.current_token() != Token::ParenOpen) // Simple variable ref.
    return std::make_unique<VariableExprAST>(loc, std::move(name));

  // This is a function call.
  lexer.consume(Token::ParenOpen);
  std::vector<std::unique_ptr<ExprAST>> args;
  if (lexer.current_token() != Token::ParenClose) {
    while (true) {
      if (auto arg = parse_expression()) args.push_back(std::move(arg));
      else return nullptr;

      if (lexer.current_token() == Token::ParenClose) break;

      if (lexer.current_token() != Token::Comma)
        return parse_error<ExprAST>(", or )", "in argument list");
      lexer.consume(Token::Comma);
    }
  }
  lexer.consume(Token::ParenClose);

  // It can be a builtin call to print
  if (name == "print") {
    if (args.size() != 1)
      return parse_error<ExprAST>("<single arg>", "as argument to print()");

    return std::make_unique<PrintExprAST>(loc, std::move(args[0]));
  }

  // Call to a user-defined function
  return std::make_unique<CallExprAST>(loc, std::move(name), std::move(args));
}

std::unique_ptr<ExprAST> Parser::parse_primary() {
  switch (lexer.current_token()) {
  default:
    fmt::println(
        "unknown token '{}' when expecting an expression",
        std::to_underlying(lexer.current_token())
    );
    return nullptr;
  case Token::Identifier: return parse_identifier_expr();
  case Token::Number: return parse_number_expr();
  case Token::ParenOpen: return parse_paren_expr();
  case Token::SqrBracketOpen: return parse_tensor_literal_expr();
  case Token::Semicolon:
  case Token::CurlyBracketClose: return nullptr;
  }
}

std::unique_ptr<ExprAST>
Parser::parse_bin_op_rhs(int expr_prec, std::unique_ptr<ExprAST> lhs) {
  // If this is a binop, find its precedence.
  while (true) {
    int tok_prec = get_tok_precedence();

    // If this is a binop that binds at least as tightly as the current binop,
    // consume it, otherwise we are done.
    if (tok_prec < expr_prec) return lhs;

    // Okay, we know this is a binop.
    Token bin_op = lexer.current_token();
    lexer.consume(bin_op);
    Location loc = lexer.last_location();

    // Parse the primary expression after the binary operator.
    auto rhs = parse_primary();
    if (!rhs)
      return parse_error<ExprAST>("expression", "to complete binary operator");

    // If BinOp binds less tightly with rhs than the operator after rhs, let
    // the pending operator take rhs as its lhs.
    int next_prec = get_tok_precedence();
    if (tok_prec < next_prec) {
      rhs = parse_bin_op_rhs(tok_prec + 1, std::move(rhs));
      if (!rhs) return nullptr;
    }

    // Merge lhs/RHS.
    u8 op = ctl::lossless_cast<u8>(std::to_underlying(bin_op));
    lhs   = std::make_unique<BinaryExprAST>(
        loc, op, std::move(lhs), std::move(rhs)
    );
  }
}

std::unique_ptr<ExprAST> Parser::parse_expression() {
  auto lhs = parse_primary();
  if (!lhs) return nullptr;

  return parse_bin_op_rhs(0, std::move(lhs));
}

std::optional<VarType> Parser::parse_type() {
  if (lexer.current_token() != Token::AngleBracketOpen)
    return parse_error_opt<VarType>("<", "to begin type");
  lexer.consume(Token::AngleBracketOpen);

  VarType type;
  while (lexer.current_token() == Token::Number) {
    type.shape.push_back(ctl::lossless_cast<i64>(lexer.value()));
    lexer.consume(Token::Number);
    if (lexer.current_token() == Token::Comma) lexer.consume(Token::Comma);
  }

  if (lexer.current_token() != Token::AngleBracketClose)
    return parse_error_opt<VarType>(">", "to end type");
  lexer.consume(Token::AngleBracketClose);
  return type;
}

std::unique_ptr<VarDeclExprAST> Parser::parse_declaration() {
  if (lexer.current_token() != Token::Let)
    return parse_error<VarDeclExprAST>("let", "to begin declaration");
  Location loc = lexer.last_location();
  lexer.consume(Token::Let);

  if (lexer.current_token() != Token::Identifier)
    return parse_error<VarDeclExprAST>("identifier", "after 'var' declaration");

  std::string id = lexer.take_identifier();
  lexer.consume(Token::Identifier);

  VarType type; // Type is optional, it can be inferred
  if (lexer.current_token() == Token::Colon) {
    lexer.consume(Token::Colon);
    auto type_opt = parse_type();
    if (!type_opt) return nullptr;
    type = *type_opt;
  }

  lexer.consume(Token::Assignment);
  auto expr = parse_expression();
  if (!expr) return nullptr;
  return std::make_unique<VarDeclExprAST>(
      loc, std::move(id), std::move(type), std::move(expr)
  );
}

std::optional<ExprASTList> Parser::parse_block() {
  if (lexer.current_token() != Token::CurlyBracketOpen)
    return parse_error_opt<ExprASTList>("{", "to begin block");
  lexer.consume(Token::CurlyBracketOpen);

  ExprASTList expr_list;

  // Ignore empty expressions: swallow sequences of semicolons.
  while (lexer.current_token() == Token::Semicolon)
    lexer.consume(Token::Semicolon);

  while (lexer.current_token() != Token::CurlyBracketClose
         && lexer.current_token() != Token::EndOfFile) {
    switch (lexer.current_token()) {
    case Token::Let: {
      // Variable declaration
      auto varDecl = parse_declaration();
      if (!varDecl) return std::nullopt;
      expr_list.push_back(std::move(varDecl));
      break;
    }
    case Token::Return: {
      // Return statement
      auto ret = parse_return();
      if (!ret) return std::nullopt;
      expr_list.push_back(std::move(ret));
      break;
    }
    default: {
      // General expression
      auto expr = parse_expression();
      if (!expr) return std::nullopt;
      expr_list.push_back(std::move(expr));
      break;
    }
    }

    // Ensure that elements are separated by a semicolon.
    if (lexer.current_token() != Token::Semicolon)
      return parse_error_opt<ExprASTList>(";", "after expression");

    // Ignore empty expressions: swallow sequences of semicolons.
    while (lexer.current_token() == Token::Semicolon)
      lexer.consume(Token::Semicolon);
  }

  if (lexer.current_token() != Token::CurlyBracketClose)
    return parse_error_opt<ExprASTList>("}", "to close block");

  lexer.consume(Token::CurlyBracketClose);
  return expr_list;
}

std::unique_ptr<PrototypeAST> Parser::parse_prototype() {
  Location proto_loc = lexer.last_location();

  if (lexer.current_token() != Token::Fn)
    return parse_error<PrototypeAST>("fn", "in prototype");
  lexer.consume(Token::Fn);

  if (lexer.current_token() != Token::Identifier)
    return parse_error<PrototypeAST>("function name", "in prototype");

  std::string fn_name = lexer.take_identifier();
  lexer.consume(Token::Identifier);

  if (lexer.current_token() != Token::ParenOpen)
    return parse_error<PrototypeAST>("(", "in prototype");
  lexer.consume(Token::ParenOpen);

  std::vector<std::unique_ptr<VariableExprAST>> args;
  if (lexer.current_token() != Token::ParenClose) {
    while (true) {
      std::string name(lexer.take_identifier());
      Location    arg_loc = lexer.last_location();
      lexer.consume(Token::Identifier);
      args.push_back(std::make_unique<VariableExprAST>(arg_loc, name));
      if (lexer.current_token() != Token::Comma) break;
      lexer.consume(Token::Comma);
      if (lexer.current_token() != Token::Identifier) {
        return parse_error<PrototypeAST>(
            "identifier", "after ',' in function parameter list"
        );
      }
    }
  }
  if (lexer.current_token() != Token::ParenClose)
    return parse_error<PrototypeAST>(")", "to end function prototype");

  // success.
  lexer.consume(Token::ParenClose);
  return std::make_unique<PrototypeAST>(
      proto_loc, std::move(fn_name), std::move(args)
  );
}

std::unique_ptr<ModuleAST> Parser::parse_module() {
  lexer.next_token(); // prime the lexer

  // Parse functions one at a time and accumulate in this vector.
  std::vector<FunctionAST> functions;
  while (auto f = parse_definition()) {
    functions.push_back(std::move(*f));
    if (lexer.current_token() == Token::EndOfFile) break;
  }
  // If we didn't reach EOF, there was an error during parsing
  if (lexer.current_token() != Token::EndOfFile)
    return parse_error<ModuleAST>("nothing", "at end of module");

  return std::make_unique<ModuleAST>(std::move(functions));
}

std::optional<FunctionAST> Parser::parse_definition() {
  auto proto = parse_prototype();
  if (!proto) return std::nullopt;

  if (auto block = parse_block())
    return std::make_optional<FunctionAST>(std::move(proto), *std::move(block));
  return std::nullopt;
}

int Parser::get_tok_precedence() {
  i32 tok = std::to_underlying(lexer.current_token());
  if (!::isascii(tok)) return -1;

  // 1 is lowest precedence.
  switch (ctl::lossless_cast<char>(tok)) {
  case '-':
  case '+': return 20;
  case '*': return 40;
  default: return -1;
  }
}

template<typename R, typename T, typename U>
std::unique_ptr<R> Parser::parse_error(T&& expected, U&& context) {
  error(std::forward<T>(expected), std::forward<U>(context));
  return {};
}

template<typename R, typename T, typename U>
std::optional<R> Parser::parse_error_opt(T&& expected, U&& context) {
  error(std::forward<T>(expected), std::forward<U>(context));
  return {};
}

template<typename T, typename U>
void Parser::error(T&& expected, U&& context) {
  i32 cur_tok = std::to_underlying(lexer.current_token());
  fmt::print(
      "Parse error ({}, {}): expected '{}' {} but has Token {}",
      lexer.last_location().line,
      lexer.last_location().col,
      expected,
      context,
      cur_tok
  );
  if (std::isprint(cur_tok))
    fmt::print(" '{}'", ctl::lossless_cast<char>(cur_tok));
  fmt::print("\n");
}
