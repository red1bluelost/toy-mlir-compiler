#ifndef TOY_PARSER_PARSER_HPP
#define TOY_PARSER_PARSER_HPP

#include "toy/ast/ast.hpp"
#include "toy/lexer/lexer.hpp"

namespace toy {

/// This is a simple recursive parser for the Toy language. It produces a well
/// formed AST from a stream of Token supplied by the Lexer. No semantic checks
/// or symbol resolution is performed. For example, variables are referenced by
/// string and the code could reference an undeclared variable and the parsing
/// succeeds.
class Parser {
 public:
  /// Create a Parser for the supplied lexer.
  Parser(Lexer lexer) : lexer(std::move(lexer)) {}

  /// Parse a full Module. A module is a list of function definitions.
  ///
  /// module ::= definition ...
  std::unique_ptr<ModuleAST> parse_module();

 private:
  /// Parse a return statement.
  /// return ::= return ; | return expr ;
  std::unique_ptr<ReturnExprAST> parse_return();

  /// Parse a literal number.
  /// numberexpr ::= number
  std::unique_ptr<ExprAST> parse_number_expr();

  /// Parse a literal array expression.
  /// tensorLiteral ::= [ literalList ] | number
  /// literalList ::= tensorLiteral | tensorLiteral, literalList
  std::unique_ptr<ExprAST> parse_tensor_literal_expr();

  /// parenexpr ::= '(' expression ')'
  std::unique_ptr<ExprAST> parse_paren_expr();

  /// identifierexpr
  ///   ::= identifier
  ///   ::= identifier '(' expression ')'
  std::unique_ptr<ExprAST> parse_identifier_expr();

  /// primary
  ///   ::= identifierexpr
  ///   ::= numberexpr
  ///   ::= parenexpr
  ///   ::= tensorliteral
  std::unique_ptr<ExprAST> parse_primary();

  /// Recursively parse the right hand side of a binary expression, the ExprPrec
  /// argument indicates the precedence of the current binary operator.
  ///
  /// binoprhs ::= ('+' primary)*
  std::unique_ptr<ExprAST>
  parse_bin_op_rhs(int expr_prec, std::unique_ptr<ExprAST> lhs);

  /// expression::= primary binop rhs
  std::unique_ptr<ExprAST> parse_expression();

  /// type ::= < shape_list >
  /// shape_list ::= num | num , shape_list
  std::optional<VarType> parse_type();

  /// Parse a variable declaration, it starts with a `let` keyword followed
  /// by and identifier and an optional type (shape specification) before
  /// the initializer. decl ::= let identifier [ type ] = expr
  std::unique_ptr<VarDeclExprAST> parse_declaration();

  /// Parse a block: a list of expression separated by semicolons and wrapped in
  /// curly braces.
  ///
  /// block ::= { expression_list }
  /// expression_list ::= block_expr ; expression_list
  /// block_expr ::= decl | "return" | expr
  std::optional<ExprASTList> parse_block();

  /// prototype ::= fn id '(' decl_list ')'
  /// decl_list ::= identifier | identifier, decl_list
  std::unique_ptr<PrototypeAST> parse_prototype();

  /// Parse a function definition, we expect a prototype initiated with the
  /// `fn` keyword, followed by a block containing a list of expressions.
  ///
  /// definition ::= prototype block
  std::optional<FunctionAST> parse_definition();

  /// Get the precedence of the pending binary operator token.
  int get_tok_precedence();

  /// Helper function to signal errors while parsing, it takes an argument
  /// indicating the expected token and another argument giving more context.
  /// Location is retrieved from the lexer to enrich the error message.
  template<typename R, typename T, typename U = const char*>
  std::unique_ptr<R> parse_error(T&& expected, U&& context = "");
  template<typename R, typename T, typename U = const char*>
  std::optional<R> parse_error_opt(T&& expected, U&& context = "");
  template<typename T, typename U = const char*>
  void error(T&& expected, U&& context = "");

  Lexer lexer;
};

} // namespace toy

#endif // TOY_PARSER_PARSER_HPP
