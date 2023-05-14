#include "toy/lexer/lexer.hpp"

#include <llvm/ADT/StringSwitch.h>

#include <ctl/object/numerics.hpp>
#include <fmt/core.h>

#include <utility>

using namespace toy;

int Lexer::process_next_char() {
  // The current line buffer should not be empty unless it is the end of file.
  if (current_line_buffer_.empty()) return EOF;
  ++current_col_;
  auto nextchar        = current_line_buffer_.front();
  current_line_buffer_ = current_line_buffer_.substr(1);
  if (current_line_buffer_.empty())
    current_line_buffer_ = line_reader_.read_line();
  if (nextchar == '\n') {
    ++current_line_;
    current_col_ = 0;
  }
  return nextchar;
}

Token Lexer::process_token() {
  // Skip any whitespace.
  while (std::isspace(last_char_)) last_char_ = process_next_char();

  // Save the current location before reading the token characters.
  last_location_.line = current_line_;
  last_location_.col  = current_col_;

  // Identifier: [a-zA-Z][a-zA-Z0-9_]*
  if (std::isalpha(last_char_)) {
    identifier_ = ctl::lossless_cast<char>(last_char_);
    while (std::isalnum(last_char_ = process_next_char()) || last_char_ == '_')
      identifier_ += ctl::lossless_cast<char>(last_char_);

    return llvm::StringSwitch<Token>(identifier_)
        .Case("return", Token::Return)
        .Case("fn", Token::Fn)
        .Case("let", Token::Let)
        .Default(Token::Identifier);
  }

  // Number: [0-9.]+
  if (std::isdigit(last_char_) || last_char_ == '.') {
    std::string num_str;
    do {
      num_str += ctl::lossless_cast<char>(last_char_);
      last_char_ = process_next_char();
    } while (isdigit(last_char_) || last_char_ == '.');

    value_ = std::strtod(num_str.c_str(), nullptr);
    return Token::Number;
  }

  if (last_char_ == '#') {
    // Comment until end of line.
    do {
      last_char_ = process_next_char();
    } while (last_char_ != EOF && last_char_ != '\n' && last_char_ != '\r');
    if (last_char_ != EOF) return process_token();
  }

  // Check for end of file. Don't eat the EOF.
  if (last_char_ == EOF) return Token::EndOfFile;

  // Otherwise, just return the character as its ascii value.
  return Token(std::exchange(last_char_, process_next_char()));
}

void toy::dump_tokens(Lexer lexer) {
  while (true) {
    switch (lexer.next_token()) {
    case Token::EndOfFile: fmt::print("Token(EOF)"); break;
    case Token::Return: fmt::print("Token(return)"); break;
    case Token::Let: fmt::print("Token(let)"); break;
    case Token::Fn: fmt::print("Token(fn)"); break;
    case Token::Identifier:
      fmt::print("Token(id = \"{}\")", lexer.identifier());
      break;
    case Token::Number: fmt::print("Token(num = {})", lexer.value()); break;
    default:
      fmt::print(
          "Token('{}')",
          ctl::lossless_cast<char>(std::to_underlying(lexer.current_token()))
      );
      break;
    }
    fmt::println("{}", lexer.last_location());
    if (lexer.current_token() == Token::EndOfFile) return;
  }
}
