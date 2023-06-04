#include "toy/dialect/dialect.hpp"
#include "toy/dialect/mlir_gen.hpp"
#include "toy/dialect/passes.hpp"
#include "toy/lexer/lexer.hpp"
#include "toy/parser/parser.hpp"
#include "toy/util/string_view_line_reader.hpp"

#include <llvm/ADT/StringSwitch.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#include <mlir/Dialect/Affine/Passes.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/Transforms/Passes.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>
#include <mlir/Transforms/Passes.h>

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

enum class Action {
  None,
  DumpTokens,
  DumpAST,
  DumpMLIR,
  DumpMLIRAffine,
  DumpMLIRLLVM,
  DumpLLVMIR,
  RunJit
};

cl::opt<enum Action> action(
    "emit",
    cl::desc("Select the kind of output desired"),
    cl::values(
        clEnumValN(Action::DumpTokens, "tokens", "output the token dump"),
        clEnumValN(Action::DumpAST, "ast", "output the AST dump"),
        clEnumValN(Action::DumpMLIR, "mlir", "output the MLIR dump"),
        clEnumValN(
            Action::DumpMLIRAffine,
            "mlir-affine",
            "output the MLIR dump after affine lowering"
        ),
        clEnumValN(
            Action::DumpMLIRLLVM,
            "mlir-llvm",
            "output the MLIR dump after llvm lowering"
        ),
        clEnumValN(Action::DumpLLVMIR, "llvm", "output the LLVM IR dump"),
        clEnumValN(
            Action::RunJit,
            "jit",
            "JIT the code and run it by invoking the main function"
        )
    )
);

cl::opt<bool> enable_opt("opt", cl::desc("Enable optimizations"));

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

mlir::OwningOpRef<mlir::ModuleOp> main_input_mlir(
    std::unique_ptr<llvm::MemoryBuffer> file, mlir::MLIRContext& context
) {
  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<mlir::toy::ToyDialect>();

  switch (input_type) {
  case InputType::Toy: {
    toy::StringViewLineReader line_reader{file->getBuffer()};
    toy::Lexer                lexer{input_filename, line_reader};
    toy::Parser               parser(std::move(lexer));

    auto module_ast = parser.parse_module();
    if (!module_ast) return nullptr;

    return toy::mlir_gen(context, *module_ast);
  }
  case InputType::MLIR: {
    llvm::SourceMgr source_mgr;
    source_mgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
    return mlir::parseSourceFile<mlir::ModuleOp>(source_mgr, &context);
  }
  case InputType::Deduce: llvm_unreachable("deduce should not be here");
  }
  llvm_unreachable("invalid enum");
}

int main_process_mlir(mlir::ModuleOp mod) {
  mlir::PassManager pm{mod->getName()};
  if (failed(mlir::applyPassManagerCLOptions(pm))) return EXIT_FAILURE;

  bool is_lowering_to_affine = action >= Action::DumpMLIRAffine;
  bool is_lowering_to_llvm   = action >= Action::DumpMLIRLLVM;

  if (enable_opt || is_lowering_to_affine) {
    // Inline all functions into main and then delete them.
    pm.addPass(mlir::createInlinerPass());

    mlir::OpPassManager& op_pm = pm.nest<mlir::toy::FuncOp>();
    op_pm.addPass(mlir::createCanonicalizerPass());
    op_pm.addPass(mlir::toy::createShapeInferencePass());
    op_pm.addPass(mlir::createCanonicalizerPass());
    op_pm.addPass(mlir::createCSEPass());
  }

  if (is_lowering_to_affine) {
    // Partially lower the toy dialect.
    pm.addPass(mlir::toy::createLowerToAffinePass());

    // Add a few cleanups post lowering.
    mlir::OpPassManager& opt_pm = pm.nest<mlir::func::FuncOp>();
    opt_pm.addPass(mlir::createCanonicalizerPass());
    opt_pm.addPass(mlir::createCSEPass());

    // Add optimizations if enabled.
    if (enable_opt) {
      opt_pm.addPass(mlir::affine::createLoopFusionPass());
      opt_pm.addPass(mlir::affine::createAffineScalarReplacementPass());
    }
  }

  if (is_lowering_to_llvm) {
    pm.addPass(mlir::toy::createLowerToLLVMPass());
    pm.addNestedPass<mlir::LLVM::LLVMFuncOp>(
        mlir::LLVM::createDIScopeForLLVMFuncOpPass()
    );
  }

  if (failed(pm.run(mod))) return EXIT_FAILURE;
  return EXIT_SUCCESS;
}

int main_dump_llvm_ir(mlir::ModuleOp mod) {
  llvm::LLVMContext llvm_ctx;

  std::unique_ptr<llvm::Module> llvm_mod =
      mlir::translateModuleToLLVMIR(mod, llvm_ctx);
  if (!llvm_mod)
    return fmt::println(stderr, "Failed to emit LLVM IR"), EXIT_FAILURE;

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  auto tm_builder_or_err = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!tm_builder_or_err)
    return fmt::println(stderr, "Could not create JITTargetMachineBuilder"),
           EXIT_FAILURE;

  auto tm_or_err = tm_builder_or_err->createTargetMachine();
  if (!tm_or_err)
    return fmt::println(stderr, "Could not create TargetMachine"), EXIT_FAILURE;

  mlir::ExecutionEngine::setupTargetTripleAndDataLayout(
      llvm_mod.get(), tm_or_err->get()
  );

  auto opt_pipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enable_opt ? 3 : 0,
      /*sizeLevel=*/0,
      /*targetMachine=*/nullptr
  );
  if (auto err = opt_pipeline(llvm_mod.get()); err) {
    std::string err_str;
    llvm::raw_string_ostream(err_str) << err;
    fmt::println("Failed to optimize LLVM IR: {}", err_str);
    return EXIT_FAILURE;
  }

  llvm_mod->dump();
  return EXIT_SUCCESS;
}

int main_run_jit(mlir::ModuleOp mod) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  auto maybe_engine = mlir::ExecutionEngine::create(
      mod,
      {
          .transformer = mlir::makeOptimizingTransformer(
              /*optLevel=*/enable_opt ? 3 : 0,
              /*sizeLevel=*/0,
              /*targetMachine=*/nullptr
          ),
          .jitCodeGenOptLevel = std::nullopt,
      }
  );
  if (auto err = maybe_engine.takeError()) {
    std::string err_str;
    llvm::raw_string_ostream(err_str) << err;
    fmt::println("Failed to construct an execution engine: {}", err_str);
    return EXIT_FAILURE;
  }

  auto engine = std::move(*maybe_engine);

  if (auto err = engine->invokePacked("main")) {
    std::string err_str;
    llvm::raw_string_ostream(err_str) << err;
    fmt::println("JIT invocation failed: {}", err_str);
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

bool validate_arguments() {
  bool is_valid = true;

  // Deduce file type or error for bad arguments
  if (input_type == InputType::Deduce) {
    auto deduced_input_type = llvm::StringSwitch<std::optional<InputType>>(
                                  sys::path::extension(input_filename)
    )
                                  .Case(".mlir", InputType::MLIR)
                                  .Case(".toy", InputType::Toy)
                                  .Default(std::nullopt);
    if (!deduced_input_type) {
      input_type.error(llvm::Twine(
          "cannot deduce input type from filename: ", input_filename
      ));
      is_valid = false;
    }
    input_type = *deduced_input_type;
  }

  bool does_not_use_mlir = action <= Action::DumpAST;

  if (input_type == InputType::MLIR && does_not_use_mlir) {
    action.error("Cannot dump tokens or dump ast when input is mlir");
    is_valid = false;
  }

  if (enable_opt && does_not_use_mlir) {
    enable_opt.error(
        "Optimization does not run for dumping tokens or dumping ast"
    );
    is_valid = false;
  }

  return is_valid;
}

} // namespace

int main(const int argc, const char* argv[]) {
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");

  if (!validate_arguments()) return EXIT_FAILURE;

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
  case Action::DumpMLIR:
  case Action::DumpMLIRAffine:
  case Action::DumpMLIRLLVM:
  case Action::DumpLLVMIR:
  case Action::RunJit: {
    mlir::MLIRContext context;

    mlir::OwningOpRef<mlir::ModuleOp> mod =
        main_input_mlir(std::move(file), context);
    if (!mod) return EXIT_FAILURE;

    if (main_process_mlir(*mod)) return EXIT_FAILURE;

    if (action <= Action::DumpMLIRLLVM) return mod->dump(), EXIT_SUCCESS;

    mlir::registerBuiltinDialectTranslation(context);
    mlir::registerLLVMDialectTranslation(context);

    if (action == Action::DumpLLVMIR) return main_dump_llvm_ir(*mod);

    return main_run_jit(*mod);
  }
  case Action::None: {
    fmt::println("No action specified (parsing only?), use -emit=<action>");
    return EXIT_SUCCESS;
  }
  }

  llvm_unreachable("Unknown action");
}