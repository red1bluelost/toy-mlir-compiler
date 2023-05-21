#include "toy/dialect/dialect.hpp"

#include "toy/core/types.hpp"

#include <mlir/IR/FunctionImplementation.h>
#include <mlir/IR/OpImplementation.h>

#include <range/v3/view/enumerate.hpp>
#include <range/v3/view/zip.hpp>

#include <ranges>

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
  printer.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
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

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::toy::ReturnOp::verify() {
  // We know that the parent operation is a function, because of the 'HasParent'
  // trait attached to the operation definition.
  auto function = cast<FuncOp>((*this)->getParentOp());

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

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "toy/dialect/ops.cpp.inc"
