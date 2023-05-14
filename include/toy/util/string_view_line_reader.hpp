#ifndef TOY_UTIL_STRING_VIEW_LINE_READER_HPP
#define TOY_UTIL_STRING_VIEW_LINE_READER_HPP

#include "toy/core/types.hpp"

#include <span>
#include <string_view>

namespace toy {

class StringViewLineReader {
 public:
  explicit StringViewLineReader(std::string_view str) : str_(str) {}

  /// Provide one line at a time, return an empty string when reaching the end
  /// of the buffer.
  std::string_view read_line() {
    auto begin = str_.begin();
    while (!str_.empty() && str_.front() != '\0' && str_.front() != '\n')
      str_ = str_.substr(1);
    if (str_.empty() || str_.front() == '\0') return "";
    str_ = str_.substr(1);
    return {begin, str_.begin()};
  }

 private:
  std::string_view str_;
};

} // namespace toy

#endif // TOY_UTIL_STRING_VIEW_LINE_READER_HPP