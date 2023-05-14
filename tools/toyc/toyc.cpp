#include "toy/lexer/lexer.hpp"
#include "toy/parser/parser.hpp"
#include "toy/util/string_view_line_reader.hpp"

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/MemoryBuffer.h>

namespace {
namespace cl = llvm::cl;

cl::opt<std::string> input_filename(
    cl::Positional,
    cl::desc("<input toy file>"),
    cl::init("-"),
    cl::value_desc("filename")
);

enum class Action { None, DumpTokens, DumpAST };

cl::opt<enum Action> action(
    "emit",
    cl::desc("Select the kind of output desired"),
    cl::values(
        clEnumValN(Action::DumpTokens, "tokens", "output the token dump"),
        clEnumValN(Action::DumpAST, "ast", "output the AST dump")
    )
);

} // namespace

int main(const int argc, const char* argv[]) {
  cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(input_filename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return EXIT_FAILURE;
  }

  toy::StringViewLineReader line_reader{fileOrErr.get()->getBuffer()};
  toy::Lexer                lexer{input_filename, line_reader};

  switch (action) {
  case Action::DumpTokens: toy::dump_tokens(std::move(lexer)); break;
  case Action::DumpAST: {
    toy::Parser parser(std::move(lexer));
    auto        module_ast = parser.parse_module();
    if (!module_ast) return EXIT_FAILURE;
    toy::dump_module_ast(*module_ast);
    break;
  }
  case Action::None: {
    fmt::println("No action specified (parsing only?), use -emit=<action>");
    break;
  }
  }

  return EXIT_SUCCESS;
}