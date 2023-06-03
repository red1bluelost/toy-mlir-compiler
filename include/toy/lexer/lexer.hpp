#ifndef TOY_LEXER_LEXER_HPP
#define TOY_LEXER_LEXER_HPP

#include "toy/lexer/location.hpp"
#include "toy/util/line_reader.hpp"

#include <cassert>
#include <optional>

namespace toy {

// List of Token returned by the lexer.
enum class Token : i32 {
  Comma             = ',',
  Semicolon         = ';',
  Colon             = ':',
  ParenOpen         = '(',
  ParenClose        = ')',
  CurlyBracketOpen  = '{',
  CurlyBracketClose = '}',
  SqrBracketOpen    = '[',
  SqrBracketClose   = ']',
  AngleBracketOpen  = '<',
  AngleBracketClose = '>',
  Assignment        = '=',
  Dot               = '.',

  EndOfFile = -1,

  /// Commands
  Return = -2,
  Let    = -3,
  Fn     = -4,
  Struct = -5,

  /// Primary
  Identifier = -6,
  Number     = -7,
};

/// The Lexer is an abstract base class providing all the facilities that the
/// Parser expects. It goes through the stream one token at a time and keeps
/// track of the location in the file for debugging purposes.
/// It relies on a subclass to provide a `readNextLine()` method. The subclass
/// can proceed by reading the next line from the standard input or from a
/// memory mapped file.
class Lexer {
 public:
  /// Create a lexer for the given filename. The filename is kept only for
  /// debugging purposes (attaching a location to a Token).
  explicit Lexer(std::string_view filename, LineReader line_reader)
      : last_location_{filename, 0, 0}
      , line_reader_(std::move(line_reader)) {}

  Lexer(Lexer&& lexer) = default;
  ~Lexer()             = default;

  Lexer& operator=(Lexer&& lexer) = default;

  /// Look at the current token in the stream.
  [[nodiscard]] Token current_token() const { return current_token_; }

  /// Move to the next token in the stream and return it.
  Token next_token() { return current_token_ = process_token(); }

  /// Move to the next token in the stream, asserting on the current token
  /// matching the expectation.
  void consume(Token tok) {
    assert(tok == current_token_ && "consume Token mismatch expectation");
    next_token();
  }

  /// Move to the next token in the stream if token is current, otherwise return
  /// false without advancing.
  bool consume_if(Token tok) {
    return tok == current_token_ && (next_token(), true);
  }

  /// Move to next token and return the string if current is an identifier,
  /// otherwise return std::nullopt
  std::optional<std::string> consume_identifier() {
    if (Token::Identifier != current_token_) return std::nullopt;
    std::string ret_str = take_identifier();
    next_token();
    return ret_str;
  }

  /// Return the current identifier (prereq: current_token_ == Identifier)
  [[nodiscard]] std::string_view identifier() const {
    assert(current_token_ == Token::Identifier);
    return identifier_;
  }

  /// Move the current identifier (prereq: current_token_ == Identifier)
  std::string take_identifier() {
    assert(current_token_ == Token::Identifier);
    return std::move(identifier_);
  }

  /// Return the current number (prereq: current_token_ == Number)
  [[nodiscard]] f64 value() const {
    assert(current_token_ == Token::Number);
    return value_;
  }

  /// Return the location for the beginning of the current token.
  [[nodiscard]] Location last_location() const { return last_location_; }

  // Return the current line in the file.
  [[nodiscard]] i32 current_line() const { return current_line_; }

  // Return the current column in the file.
  [[nodiscard]] i32 current_col() const { return current_col_; }

 private:
  /// Return the next character from the stream. This manages the buffer for the
  /// current line and request the next line buffer to the LineReader.
  i32 process_next_char();

  ///  Return the next token from standard input.
  Token process_token();

  /// The last token read from the input.
  Token current_token_ = Token::EndOfFile;

  /// Location for `current_token`.
  Location last_location_;

  /// If the current Token is an identifier, this string contains the value.
  std::string identifier_;

  /// If the current Token is a number, this contains the value.
  f64 value_ = 0;

  /// The last value returned by process_next_char(). We need to keep it around
  /// as we always need to read ahead one character to decide when to end a
  /// token and we can't put it back in the stream after reading from it.
  i32 last_char_ = ' ';

  /// Keep track of the current line number in the input stream
  i32 current_line_ = 0;

  /// Keep track of the current column number in the input stream
  i32 current_col_ = 0;

  /// Buffer supplied by the LineReader
  std::string_view current_line_buffer_ = "\n";

  /// Input stream for lines of the program
  LineReader line_reader_;
};

void dump_tokens(Lexer lexer);

} // namespace toy

#endif // TOY_LEXER_LEXER_HPP