#include "toy/parser/parser.hpp"

#include "toy/ast/ast.hpp"

#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>

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
        return parse_error("<num> or [", "in literal expression");
      if (auto number = parse_number_expr())
        values.push_back(std::move(number));
      else return nullptr; // parse error in the nested array.
    }

    // End of this list on ']'
    if (lexer.consume_if(Token::SqrBracketClose)) break;

    // Elements are separated by a comma.
    if (!lexer.consume_if(Token::Comma))
      return parse_error("] or ,", "in literal expression");
  }
  if (values.empty())
    return parse_error("<something>", "to fill literal expression");

  // Fill in the dimensions now. First the current nesting level:
  dims.push_back(ctl::lossless_cast<i64>(values.size()));

  // If there is any nested array, process all of them and ensure that
  // dimensions are uniform.
  if (contains_literals) {
    auto* first_literal = llvm::dyn_cast<LiteralExprAST>(values.front().get());
    if (!first_literal)
      return parse_error(
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
      return parse_error(
          "uniform well-nested dimensions", "inside literal expression"
      );
  }

  return std::make_unique<LiteralExprAST>(
      loc, std::move(values), std::move(dims)
  );
}

std::unique_ptr<ExprAST> Parser::parse_struct_literal_expr() {
  Location loc = lexer.last_location();
  lexer.consume(Token::CurlyBracketOpen);

  if (lexer.consume_if(Token::CurlyBracketClose))
    return parse_error("<something>", "to fill struct literal");

  std::vector<std::unique_ptr<ExprAST>> args;
  while (true) {
    switch (lexer.current_token()) {
    case Token::SqrBracketOpen:
      if (auto e = parse_tensor_literal_expr()) args.push_back(std::move(e));
      else return nullptr;
      break;
    case Token::Number:
      if (auto e = parse_number_expr()) args.push_back(std::move(e));
      else return nullptr;
      break;
    case Token::CurlyBracketOpen:
      if (auto e = parse_struct_literal_expr()) args.push_back(std::move(e));
      else return nullptr;
      break;
    default: return parse_error("{, [, or number", "in struct literal");
    }

    if (!lexer.consume_if(Token::Comma)) break;
    if (lexer.current_token() == Token::CurlyBracketClose) break;
  }
  if (!lexer.consume_if(Token::CurlyBracketClose))
    return parse_error('}', "at end of struct literal");

  return std::make_unique<StructLitExprAST>(loc, std::move(args));
}

std::unique_ptr<ExprAST> Parser::parse_paren_expr() {
  lexer.consume(Token::ParenOpen);
  auto expr = parse_expression();
  if (!expr) return nullptr;

  if (!lexer.consume_if(Token::ParenClose))
    return parse_error(")", "to close expression with parentheses");
  return expr;
}

std::unique_ptr<ExprAST> Parser::parse_identifier_expr() {
  std::string name = lexer.take_identifier();
  Location    loc  = lexer.last_location();
  lexer.consume(Token::Identifier);

  if (!lexer.consume_if(Token::ParenOpen)) // Simple variable ref.
    return std::make_unique<VariableExprAST>(loc, std::move(name));

  // This is a function call.
  std::vector<std::unique_ptr<ExprAST>> args;
  while (lexer.current_token() != Token::ParenClose) {
    if (auto arg = parse_expression()) args.push_back(std::move(arg));
    else return nullptr;

    if (lexer.consume_if(Token::ParenClose)) break;

    if (!lexer.consume_if(Token::Comma))
      return parse_error(", or )", "in argument list");
  }

  // It can be a builtin call to print
  if (name == "print") {
    if (args.size() != 1)
      return parse_error("<single arg>", "as argument to print()");

    return std::make_unique<PrintExprAST>(loc, std::move(args[0]));
  }

  // Call to a user-defined function
  return std::make_unique<CallExprAST>(loc, std::move(name), std::move(args));
}

std::unique_ptr<ExprAST> Parser::parse_primary() {
  switch (lexer.current_token()) {
  default:
    fmt::println(
        "unexpected token '{}' when expecting an expression",
        std::to_underlying(lexer.current_token())
    );
    return nullptr;
  case Token::Identifier: return parse_identifier_expr();
  case Token::Number: return parse_number_expr();
  case Token::ParenOpen: return parse_paren_expr();
  case Token::SqrBracketOpen: return parse_tensor_literal_expr();
  case Token::CurlyBracketOpen: return parse_struct_literal_expr();
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
    if (!rhs) return parse_error("expression", "to complete binary operator");

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
  switch (lexer.current_token()) {
  default: return parse_error_opt("<", "to begin type");
  case Token::AngleBracketOpen: {
    lexer.consume(Token::AngleBracketOpen);

    VarType type{VarType::ShapeVec{}};
    while (lexer.current_token() == Token::Number) {
      std::get<VarType::ShapeVec>(type.internal)
          .push_back(ctl::lossless_cast<i64>(lexer.value()));
      lexer.consume(Token::Number);
      lexer.consume_if(Token::Comma);
    }

    if (!lexer.consume_if(Token::AngleBracketClose))
      return parse_error_opt(">", "to end type");
    return type;
  }
  case Token::Identifier: {
    VarType type{lexer.take_identifier()};
    lexer.consume(Token::Identifier);
    return type;
  }
  }
}

std::unique_ptr<VarDeclExprAST> Parser::parse_declaration() {
  Location loc = lexer.last_location();
  if (!lexer.consume_if(Token::Let))
    return parse_error("let", "to begin declaration");

  std::string id;
  if (auto str_opt = lexer.consume_identifier()) id = *std::move(str_opt);
  else return parse_error("identifier", "after 'let' declaration");

  VarType type; // Type is optional, it can be inferred
  if (lexer.consume_if(Token::Colon)) {
    if (auto type_opt = parse_type()) type = *std::move(type_opt);
    else return nullptr;
  }

  lexer.consume(Token::Assignment);
  auto expr = parse_expression();
  if (!expr) return nullptr;
  return std::make_unique<VarDeclExprAST>(
      loc, std::move(id), std::move(type), std::move(expr)
  );
}

std::optional<ExprASTList> Parser::parse_block() {
  if (!lexer.consume_if(Token::CurlyBracketOpen))
    return parse_error_opt("{", "to begin block");

  // Ignore empty expressions: swallow sequences of semicolons.
  while (lexer.consume_if(Token::Semicolon)) lexer.consume(Token::Semicolon);

  ExprASTList expr_list;
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
    if (!lexer.consume_if(Token::Semicolon))
      return parse_error_opt(";", "after expression");

    // Ignore empty expressions: swallow sequences of semicolons.
    while (lexer.consume_if(Token::Semicolon)) continue;
  }

  if (!lexer.consume_if(Token::CurlyBracketClose))
    return parse_error_opt("}", "to close block");

  return expr_list;
}

std::optional<std::vector<VarDeclExprAST>>
Parser::parse_field_list(Token separator, bool requires_last) {
  std::vector<VarDeclExprAST> args;

  bool inside_identifier = false;
  while (lexer.current_token() == Token::Identifier) {
    inside_identifier = true;

    std::string name    = lexer.take_identifier();
    Location    arg_loc = lexer.last_location();
    lexer.consume(Token::Identifier);

    VarType var_type = {};
    if (lexer.consume_if(Token::Colon)) {
      if (auto vto = parse_type()) var_type = *std::move(vto);
      else return std::nullopt;
      if (!var_type.is_name())
        return parse_error_opt("struct name", "in parameters list");
    }

    args.emplace_back(arg_loc, name, var_type);
    if (!lexer.consume_if(separator)) break;

    inside_identifier = false;
  }
  if (requires_last && inside_identifier)
    return parse_error_opt("separator", "at end of parameters list");

  return args;
}

std::unique_ptr<PrototypeAST> Parser::parse_prototype() {
  Location proto_loc = lexer.last_location();
  if (!lexer.consume_if(Token::Fn)) return parse_error("fn", "in prototype");

  std::string fn_name;
  if (auto str_opt = lexer.consume_identifier()) fn_name = *std::move(str_opt);
  else return parse_error("function name", "in prototype");

  if (!lexer.consume_if(Token::ParenOpen))
    return parse_error("(", "in prototype");

  std::vector<VarDeclExprAST> args;
  if (auto args_opt = parse_field_list(Token::Comma, false))
    args = *std::move(args_opt);
  else return nullptr;
  if (!lexer.consume_if(Token::ParenClose))
    return parse_error(")", "to end function prototype");

  return std::make_unique<PrototypeAST>(
      proto_loc, std::move(fn_name), std::move(args)
  );
}

std::optional<FunctionAST> Parser::parse_func_def() {
  auto proto = parse_prototype();
  if (!proto) return std::nullopt;

  if (auto block = parse_block())
    return std::make_optional<FunctionAST>(std::move(proto), *std::move(block));
  return std::nullopt;
}

std::optional<StructAST> Parser::parse_struct_def() {
  Location loc = lexer.last_location();
  if (!lexer.consume_if(Token::Struct))
    return parse_error_opt("struct", "in struct definition");

  std::string struct_name;
  if (auto str_opt = lexer.consume_identifier())
    struct_name = *std::move(str_opt);
  else return parse_error_opt("identifier", "in struct definition");

  if (!lexer.consume_if(Token::CurlyBracketOpen))
    return parse_error_opt('{', "after struct name");

  std::vector<VarDeclExprAST> fields;
  if (auto fields_opt = parse_field_list(Token::Semicolon, true))
    fields = *std::move(fields_opt);
  else return std::nullopt;
  if (!lexer.consume_if(Token::CurlyBracketClose))
    return parse_error_opt("}", "to end struct definition");

  if (fields.empty())
    return parse_error_opt("<something>", "to fill struct definition");

  return std::make_optional<StructAST>(
      loc, std::move(struct_name), std::move(fields)
  );
}

std::unique_ptr<ModuleAST> Parser::parse_module() {
  lexer.next_token(); // prime the lexer

  if (lexer.current_token() == Token::EndOfFile)
    return parse_error("code", "in the file");

  std::vector<StructAST>   structs;
  std::vector<FunctionAST> functions;

  // Parse functions one at a time and accumulate in this vector.
  while (lexer.current_token() != Token::EndOfFile) {
    switch (lexer.current_token()) {
    case Token::Fn:
      if (auto f = parse_func_def()) functions.push_back(std::move(*f));
      else return parse_error("function", "after fn token");
      break;
    case Token::Struct:
      if (auto s = parse_struct_def()) structs.push_back(std::move(*s));
      else return parse_error("struct", "after struct token");
      break;
    default:
      return parse_error(
          "function|struct|<nothing>", "after previous definition"
      );
    }
  }
  // If we didn't reach EOF, there was an error during parsing
  if (lexer.current_token() != Token::EndOfFile)
    return parse_error("nothing", "at end of module");

  return std::make_unique<ModuleAST>(std::move(structs), std::move(functions));
}

i32 Parser::get_tok_precedence() {
  i32 tok = std::to_underlying(lexer.current_token());
  if (!::isascii(tok)) return -1;

  // 1 is lowest precedence.
  switch (ctl::lossless_cast<char>(tok)) {
  case '-':
  case '+': return 20;
  case '*': return 40;
  case '.': return 60;
  default: return -1;
  }
}

template<typename T, typename U>
std::nullptr_t Parser::parse_error(T&& expected, U&& context) {
  error(std::forward<T>(expected), std::forward<U>(context));
  return nullptr;
}

template<typename T, typename U>
std::nullopt_t Parser::parse_error_opt(T&& expected, U&& context) {
  error(std::forward<T>(expected), std::forward<U>(context));
  return std::nullopt;
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
