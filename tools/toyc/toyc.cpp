#include "toy/dialect/dialect.hpp"
#include "toy/dialect/mlir_gen.hpp"
#include "toy/lexer/lexer.hpp"
#include "toy/parser/parser.hpp"
#include "toy/util/string_view_line_reader.hpp"

#include <llvm/ADT/StringSwitch.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/SourceMgr.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Parser/Parser.h>

#include <range/v3/algorithm/contains.hpp>

#include <algorithm>
#include <optional>

namespace {
namespace cl  = llvm::cl;
namespace sys = llvm::sys;

cl::opt<std::string> input_filename(
    cl::Positional,
    cl::desc("<input toy file>"),
    cl::init("-"),
    cl::value_desc("filename")
);

enum class InputType { Deduce, Toy, MLIR };

cl::opt<InputType> input_type(
    "x",
    cl::init(InputType::Deduce),
    cl::desc("Decided the kind of input desired"),
    cl::values(
        clEnumValN(
            InputType::Deduce,
            "-",
            "deduce input type from file extension, must use file"
        ),
        clEnumValN(InputType::Toy, "toy", "load input as a Toy source"),
        clEnumValN(InputType::MLIR, "mlir", "load input as an MLIR source")
    )
);

enum class Action { None, DumpTokens, DumpAST, DumpMLIR };

cl::opt<enum Action> action(
    "emit",
    cl::desc("Select the kind of output desired"),
    cl::values(
        clEnumValN(Action::DumpTokens, "tokens", "output the token dump"),
        clEnumValN(Action::DumpAST, "ast", "output the AST dump"),
        clEnumValN(Action::DumpMLIR, "mlir", "output the MLIR dump")
    )
);

int main_dump_tokens(std::unique_ptr<llvm::MemoryBuffer> file) {
  toy::StringViewLineReader line_reader{file->getBuffer()};
  toy::Lexer                lexer{input_filename, line_reader};
  toy::dump_tokens(std::move(lexer));
  return EXIT_SUCCESS;
}

int main_dump_ast(std::unique_ptr<llvm::MemoryBuffer> file) {
  toy::StringViewLineReader line_reader{file->getBuffer()};
  toy::Lexer                lexer{input_filename, line_reader};
  toy::Parser               parser(std::move(lexer));

  auto module_ast = parser.parse_module();
  if (!module_ast) return EXIT_FAILURE;

  toy::dump_module_ast(*module_ast);
  return EXIT_SUCCESS;
}

int main_dump_mlir(std::unique_ptr<llvm::MemoryBuffer> file) {
  mlir::MLIRContext context;
  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<mlir::toy::ToyDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> mod;
  switch (input_type) {
  case InputType::Toy: {
    toy::StringViewLineReader line_reader{file->getBuffer()};
    toy::Lexer                lexer{input_filename, line_reader};
    toy::Parser               parser(std::move(lexer));
    auto                      module_ast = parser.parse_module();
    if (!module_ast) return EXIT_FAILURE;
    mod = toy::mlir_gen(context, *module_ast);
    break;
  }
  case InputType::MLIR: {
    llvm::SourceMgr source_mgr;
    source_mgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
    mod = mlir::parseSourceFile<mlir::ModuleOp>(source_mgr, &context);
    break;
  }
  case InputType::Deduce: llvm_unreachable("deduce should not be here");
  }
  if (!mod) return EXIT_FAILURE;
  mod->dump();
  return EXIT_SUCCESS;
}

} // namespace

int main(const int argc, const char* argv[]) {
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");

  // Deduce file type or error for bad arguments
  if (input_type == InputType::Deduce) {
    auto deduced_input_type = llvm::StringSwitch<std::optional<InputType>>(
                                  sys::path::extension(input_filename)
    )
                                  .Case(".mlir", InputType::MLIR)
                                  .Case(".toy", InputType::Toy)
                                  .Default(std::nullopt);
    if (!deduced_input_type.has_value()) {
      input_type.error(llvm::Twine(
          "cannot deduce input type from filename: ", input_filename
      ));
      return EXIT_FAILURE;
    }
    input_type = *deduced_input_type;
  }

  if (input_type == InputType::MLIR
      && ranges::contains(
          llvm::ArrayRef{Action::DumpTokens, Action::DumpAST}, action
      )) {
    action.error("Cannot dump tokens or dump ast when input is mlir");
    return EXIT_FAILURE;
  }

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> file_or_err =
      llvm::MemoryBuffer::getFileOrSTDIN(input_filename);
  if (std::error_code ec = file_or_err.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return EXIT_FAILURE;
  }

  std::unique_ptr<llvm::MemoryBuffer> file = std::move(*file_or_err);
  switch (action) {
  case Action::DumpTokens: return main_dump_tokens(std::move(file));
  case Action::DumpAST: return main_dump_ast(std::move(file));
  case Action::DumpMLIR: return main_dump_mlir(std::move(file));
  case Action::None: {
    fmt::println("No action specified (parsing only?), use -emit=<action>");
    return EXIT_SUCCESS;
  }
  }

  llvm_unreachable("Unknown action");
}