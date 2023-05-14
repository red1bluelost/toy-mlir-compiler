#ifndef TOY_UTIL_LINE_READER_HPP
#define TOY_UTIL_LINE_READER_HPP

#include "toy/core/types.hpp"

#include <memory>
#include <string_view>
#include <utility>

namespace toy {

template<typename R>
concept LineReadable = requires(R& r) {
  { r.read_line() } -> std::same_as<std::string_view>;
};

class LineReader {
 public:
  template<LineReadable R>
  LineReader(R r)
      : internal(std::make_unique<InternalReader<R>>(std::move(r))) {}

  std::string_view read_line() { return internal->read_line(); }

 private:
  struct IReader {
    virtual ~IReader()                   = default;
    virtual std::string_view read_line() = 0;
  };

  template<LineReadable R>
  struct InternalReader final : IReader {
    explicit InternalReader(R r) : internal(std::move(r)) {}

    std::string_view read_line() { return internal.read_line(); }

    R internal;
  };

  std::unique_ptr<IReader> internal;
};

} // namespace toy

#endif // TOY_UTIL_LINE_READER_HPP