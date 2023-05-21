#include "toy/dialect/mlir_gen.hpp"

#include "toy/ast/ast.hpp"
#include "toy/dialect/dialect.hpp"
#include "toy/lexer/location.hpp"

#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Verifier.h>

#include <ctl/object/numerics.hpp>
#include <range/v3/view/zip.hpp>

#include <numeric>
#include <string_view>

namespace {

/// Implementation of a simple MLIR emission from the Toy AST.
class MLIRGenImpl {
 private:
  using SymbolTable = llvm::ScopedHashTable<llvm::StringRef, mlir::Value>;
  using SymbolTableScope =
      llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value>;

 public:
  explicit MLIRGenImpl(mlir::MLIRContext& context) : builder_(&context) {}

  /// Public API: convert the AST for a Toy module (source file) to an MLIR
  /// Module operation.
  mlir::ModuleOp gen(const toy::ModuleAST& module_ast) {
    // We create an empty MLIR module and codegen functions one at a time and
    // add them to the module.
    mlir::ModuleOp mod = mlir::ModuleOp::create(builder_.getUnknownLoc());

    for (const toy::FunctionAST& f : module_ast.functions())
      gen(f, mod.getBody());

    if (failed(verify(mod))) {
      mod.emitError("module verification error");
      return nullptr;
    }

    return mod;
  }

 private:
  /// Converts a Toy AST location to an MLIR location.
  mlir::Location loc(const toy::Location& loc) {
    return mlir::FileLineColLoc::get(
        builder_.getStringAttr(loc.file),
        ctl::lossless_cast<unsigned>(loc.line),
        ctl::lossless_cast<unsigned>(loc.col)
    );
  }

  /// Declare a variable in the current scope, return success if the variable
  /// wasn't declared yet.
  mlir::LogicalResult declare(llvm::StringRef var, mlir::Value value) {
    return symbol_table_.count(var) == 0
             ? (symbol_table_.insert(var, value), mlir::success())
             : mlir::failure();
  }

  /// Build a tensor type from a list of shape dimensions.
  mlir::Type type_of(std::span<const toy::i64> shape) {
    // If the shape is empty, then this type is unranked.
    if (shape.empty())
      return mlir::UnrankedTensorType::get(builder_.getF64Type());

    // Otherwise, we use the given shape.
    return mlir::RankedTensorType::get(
        llvm::ArrayRef(shape.data(), shape.size()), builder_.getF64Type()
    );
  }

  /// Build an MLIR type from a Toy AST variable type (forward to the generic
  /// getType above).
  mlir::Type type_of(const toy::VarType& type) { return type_of(type.shape); }

  /// Recursive helper function to accumulate the data that compose an array
  /// literal. It flattens the nested structure in the supplied vector. For
  /// example with this array:
  ///  [[1, 2], [3, 4]]
  /// we will generate:
  ///  [ 1, 2, 3, 4 ]
  /// Individual numbers are represented as doubles.
  /// Attributes are the way MLIR attaches constant to operations.
  void collect_data(const toy::ExprAST& expr, std::vector<double>& out) {
    return llvm::TypeSwitch<const toy::ExprAST*, void>(&expr)
        .Case([&](const toy::NumberExprAST* num) {
          out.push_back(num->value());
        })
        .Case([&](const toy::LiteralExprAST* lit) {
          for (auto& value : lit->values()) collect_data(*value, out);
        })
        .Default([](auto) {
          llvm_unreachable("expected literal or number expr");
        });
  }

  /// Create the prototype for an MLIR function with as many arguments as the
  /// provided Toy AST prototype.
  mlir::toy::FuncOp gen(const toy::PrototypeAST& proto) {
    mlir::Location location = loc(proto.location());

    // This is a generic function, the return type will be inferred later.
    // Arguments type are uniformly unranked tensors.
    llvm::SmallVector<mlir::Type, 4> argTypes(
        proto.args().size(), type_of(toy::VarType{})
    );
    mlir::FunctionType funcType =
        builder_.getFunctionType(argTypes, std::nullopt);
    return builder_.create<mlir::toy::FuncOp>(location, proto.name(), funcType);
  }

  /// Emit a new function and add it to the MLIR module.
  mlir::toy::FuncOp
  gen(const toy::FunctionAST& func_ast, mlir::Block* mod_block) {
    // Create a scope in the symbol table to hold variable declarations.
    SymbolTableScope var_scope{symbol_table_};

    // Create an MLIR function for the given prototype.
    builder_.setInsertionPointToEnd(mod_block);
    mlir::toy::FuncOp function = gen(func_ast.proto());
    if (!function) return nullptr;

    // Let's start the body of the function now!
    mlir::Block& entryBlock = function.front();

    // Declare all the function arguments in the symbol table.
    for (auto [name, value] :
         ranges::zip_view(func_ast.proto().args(), entryBlock.getArguments())) {
      if (failed(declare(name->name(), value))) return nullptr;
    }

    // Set the insertion point in the builder to the beginning of the function
    // body, it will be used throughout the codegen to create operations in this
    // function.
    builder_.setInsertionPointToStart(&entryBlock);

    // Emit the body of the function.
    if (failed(gen(func_ast.body()))) {
      function.erase();
      return nullptr;
    }

    // Implicitly return void if no return statement was emitted.
    // FIXME: we may fix the parser instead to always return the last expression
    // (this would possibly help the REPL case later)
    mlir::toy::ReturnOp return_op;
    if (!entryBlock.empty())
      return_op = llvm::dyn_cast<mlir::toy::ReturnOp>(entryBlock.back());
    if (!return_op) {
      builder_.create<mlir::toy::ReturnOp>(loc(func_ast.proto().location()));
    } else if (return_op.hasOperand()) {
      // Otherwise, if this return operation has an operand then add a result to
      // the function.
      function.setType(builder_.getFunctionType(
          function.getFunctionType().getInputs(), type_of(toy::VarType{})
      ));
    }

    return function;
  }

  /// Codegen a list of expression, return failure if one of them hit an error.
  mlir::LogicalResult gen(toy::ConstExprASTSpan block_ast) {
    SymbolTableScope var_scope{symbol_table_};
    for (auto& expr : block_ast) {
      // Specific handling for variable declarations, return statement, and
      // print. These can only appear in block list and not in nested
      // expressions.
      if (auto* vardecl = llvm::dyn_cast<toy::VarDeclExprAST>(expr.get())) {
        if (!gen(*vardecl)) return mlir::failure();
        continue;
      }
      if (auto* ret = llvm::dyn_cast<toy::ReturnExprAST>(expr.get()))
        return gen(*ret);
      if (auto* print = llvm::dyn_cast<toy::PrintExprAST>(expr.get())) {
        if (failed(gen(*print))) return mlir::success();
        continue;
      }

      // Generic expression dispatch codegen.
      if (!gen(*expr)) return mlir::failure();
    }
    return mlir::success();
  }

  /// Dispatch codegen for the right expression subclass using RTTI.
  mlir::Value gen(const toy::ExprAST& expr) {
    return llvm::TypeSwitch<const toy::ExprAST*, mlir::Value>(&expr)
        .Case<
            toy::BinaryExprAST,
            toy::VariableExprAST,
            toy::LiteralExprAST,
            toy::CallExprAST,
            toy::NumberExprAST>([&](const auto* e) { return gen(*e); })
        .Default([&](const auto* e) {
          emitError(loc(e->location()))
              << "MLIR codegen encountered an unhandled expr kind '"
              << llvm::Twine(std::to_underlying(e->getKind())) << "'";
          return nullptr;
        });
  }

  /// Handle a variable declaration, we'll codegen the expression that forms the
  /// initializer and record the value in the symbol table before returning it.
  /// Future expressions will be able to reference this variable through symbol
  /// table lookup.
  mlir::Value gen(const toy::VarDeclExprAST& vardecl) {
    mlir::Value value = gen(vardecl.init_val());
    if (!value) return nullptr;

    // We have the initializer value, but in case the variable was declared
    // with specific shape, we emit a "reshape" operation. It will get
    // optimized out later as needed.
    if (!vardecl.type().shape.empty()) {
      value = builder_.create<mlir::toy::ReshapeOp>(
          loc(vardecl.location()), type_of(vardecl.type()), value
      );
    }

    // Register the value in the symbol table.
    if (failed(declare(vardecl.name(), value))) return nullptr;
    return value;
  }

  /// This is a reference to a variable in an expression. The variable is
  /// expected to have been declared and so should have a value in the symbol
  /// table, otherwise emit an error and return nullptr.
  mlir::Value gen(const toy::VariableExprAST& expr) {
    if (auto variable = symbol_table_.lookup(expr.name())) return variable;

    emitError(loc(expr.location()), "error: unknown variable '")
        << expr.name() << "'";
    return nullptr;
  }

  /// Emit a constant for a single number (FIXME: semantic? broadcast?)
  mlir::Value gen(const toy::NumberExprAST& num) {
    return builder_.create<mlir::toy::ConstantOp>(
        loc(num.location()), num.value()
    );
  }

  /// Emit a literal/constant array. It will be emitted as a flattened array of
  /// data in an Attribute attached to a `toy.constant` operation.
  /// See documentation on [Attributes](LangRef.md#attributes) for more details.
  /// Here is an excerpt:
  ///
  ///   Attributes are the mechanism for specifying constant data in MLIR in
  ///   places where a variable is never allowed [...]. They consist of a name
  ///   and a concrete attribute value. The set of expected attributes, their
  ///   structure, and their interpretation are all contextually dependent on
  ///   what they are attached to.
  ///
  /// Example, the source level statement:
  ///   var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  /// will be converted to:
  ///   %0 = "toy.constant"() {value: dense<tensor<2x3xf64>,
  ///     [[1.000000e+00, 2.000000e+00, 3.000000e+00],
  ///      [4.000000e+00, 5.000000e+00, 6.000000e+00]]>} : () -> tensor<2x3xf64>
  ///
  mlir::Value gen(const toy::LiteralExprAST& lit) {
    mlir::Type type = type_of(lit.dims());

    // The attribute is a vector with a floating point value per element
    // (number) in the array, see `collectData()` below for more details.
    std::vector<double> data;
    auto                dims = lit.dims();
    data.reserve(std::accumulate(
        dims.begin(), dims.end(), toy::usize{1}, std::multiplies<>{}
    ));
    collect_data(lit, data);

    // The type of this attribute is tensor of 64-bit floating-point with the
    // shape of the literal.
    mlir::RankedTensorType data_type = mlir::RankedTensorType::get(
        llvm::ArrayRef(dims.data(), dims.size()), builder_.getF64Type()
    );

    // This is the actual attribute that holds the list of values for this
    // tensor literal.
    auto data_attr =
        mlir::DenseElementsAttr::get(data_type, llvm::ArrayRef(data));

    // Build the MLIR op `toy.constant`. This invokes the `ConstantOp::build`
    // method.
    return builder_.create<mlir::toy::ConstantOp>(
        loc(lit.location()), type, data_attr
    );
  }

  /// Emit a binary operation
  mlir::Value gen(const toy::BinaryExprAST& binop) {
    // First emit the operations for each side of the operation before emitting
    // the operation itself. For example if the expression is `a + foo(a)`
    // 1) First it will visiting the LHS, which will return a reference to the
    //    value holding `a`. This value should have been emitted at declaration
    //    time and registered in the symbol table, so nothing would be
    //    codegen'd. If the value is not in the symbol table, an error has been
    //    emitted and nullptr is returned.
    // 2) Then the RHS is visited (recursively) and a call to `foo` is emitted
    //    and the result value is returned. If an error occurs we get a nullptr
    //    and propagate.
    //
    mlir::Value lhs = gen(binop.lhs());
    if (!lhs) return nullptr;
    mlir::Value rhs = gen(binop.rhs());
    if (!rhs) return nullptr;
    mlir::Location location = loc(binop.location());

    // Derive the operation name from the binary operator. At the moment we only
    // support '+' and '*'.
    switch (binop.op()) {
    case '+': return builder_.create<mlir::toy::AddOp>(location, lhs, rhs);
    case '*': return builder_.create<mlir::toy::MulOp>(location, lhs, rhs);
    }

    emitError(location, "invalid binary operator '") << binop.op() << "'";
    return nullptr;
  }

  /// Emit a print expression. It emits specific operations for two builtins:
  /// transpose(x) and print(x).
  mlir::LogicalResult gen(const toy::PrintExprAST& call) {
    auto arg = gen(call.arg());
    if (!arg) return mlir::failure();

    builder_.create<mlir::toy::PrintOp>(loc(call.location()), arg);
    return mlir::success();
  }

  /// Emit a call expression. It emits specific operations for the `transpose`
  /// builtin. Other identifiers are assumed to be user-defined functions.
  mlir::Value gen(const toy::CallExprAST& call) {
    std::string_view callee   = call.callee();
    mlir::Location   location = loc(call.location());

    // Codegen the operands first.
    llvm::SmallVector<mlir::Value, 4> operands;
    for (const auto& expr : call.args()) {
      mlir::Value arg = gen(*expr);
      if (!arg) return nullptr;
      operands.push_back(arg);
    }

    // Builtin calls have their custom operation, meaning this is a
    // straightforward emission.
    if (callee == "transpose") {
      if (call.args().size() != 1) {
        emitError(
            location,
            "MLIR codegen encountered an error: toy.transpose "
            "does not accept multiple arguments"
        );
        return nullptr;
      }
      return builder_.create<mlir::toy::TransposeOp>(location, operands[0]);
    }

    // Otherwise this is a call to a user-defined function. Calls to
    // user-defined functions are mapped to a custom call that takes the callee
    // name as an attribute.
    return builder_.create<mlir::toy::GenericCallOp>(
        location, callee, operands
    );
  }

  /// Emit a return operation. This will return failure if any generation fails.
  mlir::LogicalResult gen(const toy::ReturnExprAST& ret) {
    mlir::Location location = loc(ret.location());

    // 'return' takes an optional expression, handle that case here.
    mlir::Value expr = nullptr;
    if (ret.expr()) {
      expr = gen(*ret.expr());
      if (!expr) return mlir::failure();
    }

    // Otherwise, this return operation has zero operands.
    builder_.create<mlir::toy::ReturnOp>(
        location, !expr ? llvm::ArrayRef<mlir::Value>() : expr
    );
    return mlir::success();
  }

  mlir::OpBuilder builder_;
  SymbolTable     symbol_table_;
};

} // namespace

mlir::OwningOpRef<mlir::ModuleOp>
toy::mlir_gen(mlir::MLIRContext& context, const toy::ModuleAST& module_ast) {
  return MLIRGenImpl(context).gen(module_ast);
}