#include "toy/dialect/dialect.hpp"

#include "toy/core/types.hpp"

#include <mlir/IR/FunctionImplementation.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Transforms/InliningUtils.h>

#include <range/v3/view/enumerate.hpp>
#include <range/v3/view/zip.hpp>

#include <ranges>

//===----------------------------------------------------------------------===//
// ToyInlinerInterface
//===----------------------------------------------------------------------===//

/// This class defines the interface for handling inlining with Toy
/// operations.
namespace {
struct ToyInlinerInterface final : public mlir::DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  /// All call operations within toy can be inlined.
  bool isLegalToInline(
      mlir::Operation* call, mlir::Operation* callable, bool wouldBeCloned
  ) const final {
    return true;
  }

  /// All operations within toy can be inlined.
  bool isLegalToInline(mlir::Operation*, mlir::Region*, bool, mlir::IRMapping&)
      const final {
    return true;
  }

  // All functions within toy can be inlined.
  bool isLegalToInline(mlir::Region*, mlir::Region*, bool, mlir::IRMapping&)
      const final {
    return true;
  }

  //===--------------------------------------------------------------------===//
  // Transformation Hooks
  //===--------------------------------------------------------------------===//

  /// Handle the given inlined terminator(toy.return) by replacing it with a new
  /// operation as necessary.
  void handleTerminator(
      mlir::Operation* op, mlir::ArrayRef<mlir::Value> valuesToRepl
  ) const final {
    // Only "toy.return" needs to be handled here.
    auto returnOp = cast<mlir::toy::ReturnOp>(op);

    // Replace the values directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto& [idx, val] : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[idx].replaceAllUsesWith(val);
  }

  /// Attempts to materialize a conversion for a type mismatch between a call
  /// from this dialect, and a callable region. This method should generate an
  /// operation that takes 'input' as the only operand, and produces a single
  /// result of 'resultType'. If a conversion can not be generated, nullptr
  /// should be returned.
  mlir::Operation* materializeCallConversion(
      mlir::OpBuilder& builder,
      mlir::Value      input,
      mlir::Type       resultType,
      mlir::Location   conversionLoc
  ) const final {
    return builder.create<mlir::toy::CastOp>(conversionLoc, resultType, input);
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// ToyDialect
//===----------------------------------------------------------------------===//

#include "toy/dialect/dialect.cpp.inc"

/// Dialect initialization, the instance will be owned by the context. This is
/// the point of registration of types and operations for the dialect.
void mlir::toy::ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "toy/dialect/ops.cpp.inc"
      >();
  addInterfaces<ToyInlinerInterface>();
}

//===----------------------------------------------------------------------===//
// Toy Operations
//===----------------------------------------------------------------------===//

/// A generalized parser for binary operations. This parses the different forms
/// of 'printBinaryOp' below.
static mlir::ParseResult
parse_binary_op(mlir::OpAsmParser& parser, mlir::OperationState& result) {
  // $operands[0], $operands[1] $result.attributes : $type
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 2> operands;
  mlir::SMLoc operands_loc = parser.getCurrentLocation();
  mlir::Type  type;
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/2)
      || parser.parseOptionalAttrDict(result.attributes)
      || parser.parseColonType(type))
    return mlir::failure();

  // If the type is a function type, it contains the input and result types of
  // this operation.
  if (auto func_type = type.dyn_cast<mlir::FunctionType>()) {
    if (parser.resolveOperands(
            operands, func_type.getInputs(), operands_loc, result.operands
        ))
      return mlir::failure();
    result.addTypes(func_type.getResults());
    return mlir::success();
  }

  // Otherwise, the parsed type is the type of both operands and results.
  if (parser.resolveOperands(operands, type, result.operands))
    return mlir::failure();
  result.addTypes(type);
  return mlir::success();
}

/// A generalized printer for binary operations. It prints in two different
/// forms depending on if all of the types match.
static void print_binary_op(mlir::OpAsmPrinter& printer, mlir::Operation* op) {
  printer << " " << op->getOperands();
  printer.printOptionalAttrDict(op->getAttrs());
  printer << " : ";

  // If all of the types are the same, print the type directly.
  mlir::Type result_type = *op->result_type_begin();
  if (llvm::all_of(op->getOperandTypes(), [=](mlir::Type type) {
        return type == result_type;
      })) {
    printer << result_type;
    return;
  }

  // Otherwise, print a functional type.
  printer.printFunctionalType(op->getOperandTypes(), op->getResultTypes());
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

/// Build a constant operation.
/// The builder is passed as an argument, so is the state that this method is
/// expected to fill in order to build the operation.
void mlir::toy::ConstantOp::build(
    OpBuilder& builder, OperationState& state, ::toy::f64 value
) {
  auto type = RankedTensorType::get({}, builder.getF64Type());
  auto attr = DenseElementsAttr::get(type, value);
  ConstantOp::build(builder, state, type, attr);
}

/// The 'OpAsmParser' class provides a collection of methods for parsing
/// various punctuation, as well as attributes, operands, types, etc. Each of
/// these methods returns a `ParseResult`. This class is a wrapper around
/// `LogicalResult` that can be converted to a boolean `true` value on failure,
/// or `false` on success. This allows for easily chaining together a set of
/// parser rules. These rules are used to populate an `mlir::OperationState`
/// similarly to the `build` methods described above.
mlir::ParseResult
mlir::toy::ConstantOp::parse(OpAsmParser& parser, OperationState& result) {
  // $attr $value
  DenseElementsAttr value;
  if (parser.parseOptionalAttrDict(result.attributes)
      || parser.parseAttribute(value, "value", result.attributes))
    return failure();

  result.addTypes(value.getType());
  return success();
}

/// The 'OpAsmPrinter' class is a stream that allows for formatting
/// strings, attributes, operands, types, etc.
void mlir::toy::ConstantOp::print(OpAsmPrinter& printer) {
  printer << " ";
  printer.printOptionalAttrDict(
      getOperation()->getAttrs(), /*elidedAttrs=*/{"value"}
  );
  printer << getValue();
}

/// Verifier for the constant operation.
mlir::LogicalResult mlir::toy::ConstantOp::verify() {
  // If the return type of the constant is not an unranked tensor, the shape
  // must match the shape of the attribute holding the data.
  auto result_type = dyn_cast<RankedTensorType>(getResult().getType());
  if (!result_type) return success();

  // Check that the rank of the attribute type matches the rank of the constant
  // result type.
  auto attr_type = cast<RankedTensorType>(getValue().getType());
  if (attr_type.getRank() != result_type.getRank()) {
    return emitOpError(
               "return type must match the one of the attached value "
               "attribute: "
           )
        << attr_type.getRank() << " != " << result_type.getRank();
  }

  // Check that each of the dimensions match between the two types.
  for (auto [dim, attr_res] :
       ranges::zip_view(attr_type.getShape(), result_type.getShape())
           | ranges::view::enumerate) {
    auto [attr, res] = attr_res;
    if (attr != res) {
      return emitOpError(
                 "return type shape mismatches its attribute at dimension "
             )
          << dim << ": " << attr << " != " << res;
    }
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

void mlir::toy::AddOp::build(
    OpBuilder& builder, OperationState& state, Value lhs, Value rhs
) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}

mlir::ParseResult
mlir::toy::AddOp::parse(OpAsmParser& parser, OperationState& result) {
  return parse_binary_op(parser, result);
}

void mlir::toy::AddOp::print(OpAsmPrinter& p) { print_binary_op(p, *this); }

void mlir::toy::AddOp::inferShapes() {
  getResult().setType(getLhs().getType());
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

void mlir::toy::MulOp::build(
    OpBuilder& builder, OperationState& state, Value lhs, Value rhs
) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}

mlir::ParseResult
mlir::toy::MulOp::parse(OpAsmParser& parser, OperationState& result) {
  return parse_binary_op(parser, result);
}

void mlir::toy::MulOp::print(OpAsmPrinter& p) { print_binary_op(p, *this); }

void mlir::toy::MulOp::inferShapes() {
  getResult().setType(getLhs().getType());
}

//===----------------------------------------------------------------------===//
// GenericCallOp
//===----------------------------------------------------------------------===//

void mlir::toy::GenericCallOp::build(
    OpBuilder&      builder,
    OperationState& state,
    StringRef       callee,
    ArrayRef<Value> arguments
) {
  // Generic call always returns an unranked Tensor initially.
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(arguments);
  state.addAttribute(
      "callee", SymbolRefAttr::get(builder.getContext(), callee)
  );
}

/// Return the callee of the generic call operation, this is required by the
/// call interface.
mlir::CallInterfaceCallable mlir::toy::GenericCallOp::getCallableForCallee() {
  return getOperation()->getAttrOfType<SymbolRefAttr>("callee");
}

/// Get the argument operands to the called function, this is required by the
/// call interface.
mlir::Operation::operand_range mlir::toy::GenericCallOp::getArgOperands() {
  return getInputs();
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

void mlir::toy::TransposeOp::build(
    OpBuilder& builder, OperationState& state, Value value
) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(value);
}

mlir::LogicalResult mlir::toy::TransposeOp::verify() {
  auto input_type  = dyn_cast<RankedTensorType>(getOperand().getType());
  auto result_type = dyn_cast<RankedTensorType>(getType());
  if (!input_type || !result_type) return mlir::success();

  auto input_shape = input_type.getShape();
  if (std::ranges::equal(
          input_shape, std::views::reverse(result_type.getShape())
      ))
    return mlir::success();
  return emitError() << "expected result shape to be a transpose of the input";
}

void mlir::toy::TransposeOp::inferShapes() {
  auto type = cast<RankedTensorType>(getOperand().getType());
  SmallVector<::toy::i64, 2> dims(llvm::reverse(type.getShape()));
  getResult().setType(RankedTensorType::get(dims, type.getElementType()));
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

/// Infer the output shape of the CastOp, this is required by the shape
/// inference interface.
void mlir::toy::CastOp::inferShapes() {
  getResult().setType(getInput().getType());
}

/// Returns true if the given set of input and result types are compatible with
/// this cast operation.
bool mlir::toy::CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1) return false;
  // The inputs must be Tensors with the same element type.
  TensorType input  = dyn_cast<TensorType>(inputs.front());
  TensorType output = dyn_cast<TensorType>(outputs.front());
  if (!input || !output || input.getElementType() != output.getElementType())
    return false;
  // The shape is required to match if both types are ranked.
  return !input.hasRank() || !output.hasRank() || input == output;
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::toy::ReturnOp::verify() {
  // We know that the parent operation is a function, because of the 'HasParent'
  // trait attached to the operation definition.
  auto function = cast<FuncOp>(getOperation()->getParentOp());

  /// ReturnOps can only have a single optional operand.
  if (getNumOperands() > 1)
    return emitOpError() << "expects at most 1 return operand";

  // The operand number and types must match the function signature.
  const auto& results = function.getFunctionType().getResults();
  if (getNumOperands() != results.size())
    return emitOpError() << "does not return the same number of values ("
                         << getNumOperands() << ") as the enclosing function ("
                         << results.size() << ")";

  // If the operation does not have an input, we are done.
  if (!hasOperand()) return mlir::success();

  auto input_type  = *operand_type_begin();
  auto result_type = results.front();

  // Check that the result type of the function matches the operand type.
  if (input_type == result_type || isa<mlir::UnrankedTensorType>(input_type)
      || isa<mlir::UnrankedTensorType>(result_type))
    return mlir::success();

  return emitError() << "type of return operand (" << input_type
                     << ") doesn't match function result type (" << result_type
                     << ")";
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

void mlir::toy::FuncOp::build(
    OpBuilder&               builder,
    OperationState&          state,
    StringRef                name,
    FunctionType             type,
    ArrayRef<NamedAttribute> attrs
) {
  // FunctionOpInterface provides a convenient `build` method that will populate
  // the state of our FuncOp, and create an entry block.
  buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}

mlir::ParseResult
mlir::toy::FuncOp::parse(OpAsmParser& parser, OperationState& result) {
  // Dispatch to the FunctionOpInterface provided utility method that parses the
  // function operation.
  auto buildFuncType = [](Builder&       builder,
                          ArrayRef<Type> argTypes,
                          ArrayRef<Type> results,
                          function_interface_impl::VariadicFlag,
                          std::string&) {
    return builder.getFunctionType(argTypes, results);
  };

  return function_interface_impl::parseFunctionOp(
      parser,
      result,
      /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name),
      buildFuncType,
      getArgAttrsAttrName(result.name),
      getResAttrsAttrName(result.name)
  );
}

void mlir::toy::FuncOp::print(OpAsmPrinter& p) {
  // Dispatch to the FunctionOpInterface provided utility method that prints the
  // function operation.
  function_interface_impl::printFunctionOp(
      p,
      *this,
      /*isVariadic=*/false,
      getFunctionTypeAttrName(),
      getArgAttrsAttrName(),
      getResAttrsAttrName()
  );
}

/// Returns the region on the function operation that is callable.
mlir::Region* mlir::toy::FuncOp::getCallableRegion() { return &getBody(); }

/// Returns the results types that the callable region produces when
/// executed.
llvm::ArrayRef<mlir::Type> mlir::toy::FuncOp::getCallableResults() {
  return getFunctionType().getResults();
}

/// Returns the argument attributes for all callable region arguments or
/// null if there are none.
mlir::ArrayAttr mlir::toy::FuncOp::getCallableArgAttrs() {
  return getArgAttrs().value_or(nullptr);
}

/// Returns the result attributes for all callable region results or
/// null if there are none.
mlir::ArrayAttr mlir::toy::FuncOp::getCallableResAttrs() {
  return getResAttrs().value_or(nullptr);
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "toy/dialect/ops.cpp.inc"
