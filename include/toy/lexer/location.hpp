#ifndef TOY_LEXER_LOCATION_HPP
#define TOY_LEXER_LOCATION_HPP

#include "toy/core/types.hpp"

#include <fmt/core.h>

#include <string_view>

namespace toy {

/// Location of a given element in a file.
/// This has a lifetime constrained by the owner of the filename string.
struct Location {
  std::string_view file; ///< filename
  i32              line; ///< line number
  i32              col;  ///< column number
  friend constexpr bool
  operator==(const Location&, const Location&) noexcept = default;
};

} // namespace toy

template<>
struct fmt::formatter<toy::Location> {
  template<typename ParseContext>
  constexpr auto parse(ParseContext& ctx) {
    return ctx.begin();
  }

  template<typename FormatContext>
  auto format(const toy::Location& loc, FormatContext& ctx)
      -> decltype(ctx.out()) {
    return fmt::format_to(ctx.out(), "@{}:{}:{}", loc.file, loc.line, loc.col);
  }
};

#endif // TOY_LEXER_LOCATION_HPP