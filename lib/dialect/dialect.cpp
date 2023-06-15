#include "toy/dialect/dialect.hpp"

#include "toy/core/types.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/DialectImplementation.h>
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
  bool isLegalToInline(mlir::Operation*, mlir::Operation*, bool) const final {
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
  addTypes<StructType>();
  addOperations<
#define GET_OP_LIST
#include "toy/dialect/ops.cpp.inc"
      >();
  addInterfaces<ToyInlinerInterface>();
}

mlir::Operation* mlir::toy::ToyDialect::materializeConstant(
    mlir::OpBuilder& builder,
    mlir::Attribute  value,
    mlir::Type       type,
    mlir::Location   loc
) {
  return llvm::TypeSwitch<mlir::Type, mlir::Operation*>(type)
      .Case([&](mlir::toy::StructType) {
        return builder.create<mlir::toy::StructConstantOp>(
            loc, type, llvm::cast<mlir::ArrayAttr>(value)
        );
      })
      .Default([&](auto) {
        return builder.create<mlir::toy::ConstantOp>(
            loc, type, llvm::cast<mlir::DenseElementsAttr>(value)
        );
      });
}

mlir::Type mlir::toy::ToyDialect::parseType(DialectAsmParser& parser) const {
  // Parse a struct type in the following form:
  //   struct-type ::= `struct` `<` type (`,` type)* `>`

  // Parse: `struct` `<`
  if (parser.parseKeyword("struct") || parser.parseLess()) return nullptr;

  // Parse the element types of the struct.
  llvm::SmallVector<mlir::Type, 1> element_types;
  do {
    // Parse the current element type.
    llvm::SMLoc type_location = parser.getCurrentLocation();
    mlir::Type  element_type;
    if (parser.parseType(element_type)) return nullptr;

    // Check that the type is either a TensorType or another StructType.
    if (!llvm::isa<mlir::TensorType, StructType>(element_type)) {
      parser.emitError(
          type_location,
          "element type for a struct must either be a TensorType or a "
          "StructType, got: "
      ) << element_type;
      return nullptr;
    }

    element_types.push_back(element_type);
    // Parse the optional: `,`
  } while (succeeded(parser.parseOptionalComma()));

  // Parse: `>`
  if (parser.parseGreater()) return nullptr;

  return StructType::get(element_types);
}

void mlir::toy::ToyDialect::printType(
    mlir::Type type, mlir::DialectAsmPrinter& printer
) const {
  // Currently the only toy type is a struct type.
  auto struct_type = llvm::cast<StructType>(type);

  // Print the struct type according to the parser format.
  printer << "struct<";
  llvm::interleaveComma(struct_type.getElementTypes(), printer);
  printer << '>';
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

/// Verify that the given attribute value is valid for the given type.
static mlir::LogicalResult verify_constant_for_type(
    mlir::Type type, mlir::Attribute opaque_value, mlir::Operation* op
) {
  return llvm::TypeSwitch<mlir::Type, mlir::LogicalResult>(type)
      .Case([&](mlir::TensorType) -> mlir::LogicalResult {
        auto attr_value =
            llvm::dyn_cast<mlir::DenseFPElementsAttr>(opaque_value);
        if (!attr_value) {
          return op->emitError(
                     "constant of TensorType must be initialized by a "
                     "DenseFPElementsAttr, got "
                 )
              << opaque_value;
        }

        // If the return type of the constant is not an unranked tensor, the
        // shape must match the shape of the attribute holding the data.
        auto result_type = llvm::dyn_cast<mlir::RankedTensorType>(type);
        if (!result_type) return mlir::success();

        auto attr_type =
            llvm::dyn_cast<mlir::RankedTensorType>(attr_value.getType());
        if (!attr_type) return mlir::failure();

        // Check that the rank of the attribute type matches the rank of the
        // constant result type.
        if (attr_type.getRank() != result_type.getRank()) {
          return op->emitOpError(
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
            return op->emitOpError(
                       "return type shape mismatches its attribute at "
                       "dimension "
                   )
                << dim << ": " << attr << " != " << res;
          }
        }
        return mlir::success();
      })
      .Case([&](mlir::toy::StructType struct_type) -> mlir::LogicalResult {
        auto struct_element_types = struct_type.getElementTypes();

        auto attr_value = llvm::dyn_cast<mlir::ArrayAttr>(opaque_value);
        if (!attr_value
            || attr_value.getValue().size() != struct_element_types.size()) {
          return op->emitError(
                     "constant of StructType must be initialized by an "
                     "ArrayAttr with the same number of elements, got "
                 )
              << opaque_value;
        }

        auto attr_element_values = attr_value.getValue();
        for (const auto [struct_elem_type, attr_elem_value] :
             llvm::zip(struct_element_types, attr_element_values)) {
          if (failed(verify_constant_for_type(
                  struct_elem_type, attr_elem_value, op
              )))
            return mlir::failure();
        }
        return mlir::success();
      })
      .Default([](auto) {
        llvm_unreachable("Unknown type to verify");
        return mlir::failure();
      });
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
  return verify_constant_for_type(
      getResult().getType(), getValue(), getOperation()
  );
}

void mlir::toy::ConstantOp::inferShapes() {
  getResult().setType(llvm::cast<TensorType>(getValue().getType()));
}

mlir::OpFoldResult mlir::toy::ConstantOp::fold(FoldAdaptor) {
  return getValue();
}

//===----------------------------------------------------------------------===//
// StructConstantOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::toy::StructConstantOp::verify() {
  return verify_constant_for_type(
      getResult().getType(), getValue(), getOperation()
  );
}

mlir::OpFoldResult mlir::toy::StructConstantOp::fold(FoldAdaptor) {
  return getValue();
}

//===----------------------------------------------------------------------===//
// StructAccessOp
//===----------------------------------------------------------------------===//

void mlir::toy::StructAccessOp::build(
    mlir::OpBuilder&      builder,
    mlir::OperationState& state,
    mlir::Value           input,
    ::toy::usize          index
) {
  auto struct_type = llvm::cast<mlir::toy::StructType>(input.getType());
  assert(index < struct_type.getNumElementTypes());
  mlir::Type result_type = struct_type.getElementTypes()[index];
  build(
      builder,
      state,
      result_type,
      input,
      builder.getI64IntegerAttr(static_cast<int64_t>(index))
  );
}

mlir::LogicalResult mlir::toy::StructAccessOp::verify() {
  auto struct_type = llvm::cast<mlir::toy::StructType>(getInput().getType());
  ::toy::usize index_value = getIndex();
  if (index_value >= struct_type.getNumElementTypes()) {
    return emitOpError()
        << "index should be within the range of the input struct type";
  }

  mlir::Type result_type = getResult().getType();
  if (result_type != struct_type.getElementTypes()[index_value]) {
    return emitOpError() << "must have the same result type as the struct "
                            "element referred to by the index";
  }
  return mlir::success();
}

mlir::OpFoldResult mlir::toy::StructAccessOp::fold(FoldAdaptor adaptor) {
  auto struct_attr =
      llvm::dyn_cast_or_null<mlir::ArrayAttr>(adaptor.getInput());
  if (!struct_attr) return nullptr;
  return struct_attr[static_cast<unsigned>(getIndex())];
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

/// Set the callee for the generic call operation, this is required by the call
/// interface.
void mlir::toy::GenericCallOp::setCalleeFromCallable(
    mlir::CallInterfaceCallable callee
) {
  getOperation()->setAttr("callee", callee.get<SymbolRefAttr>());
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

//===----------------------------------------------------------------------===//
// Toy Types
//===----------------------------------------------------------------------===//

namespace mlir::toy::detail {
struct StructTypeStorage : public mlir::TypeStorage {
  using KeyTy = ArrayRef<Type>;

  explicit StructTypeStorage(ArrayRef<Type> elementTypes)
      : element_types_(elementTypes) {}

  friend bool operator==(const StructTypeStorage& sts, const KeyTy& key) {
    return key == sts.element_types_;
  }

  static StructTypeStorage*
  construct(TypeStorageAllocator& allocator, const KeyTy& key) {
    // Copy the elements from the provided `KeyTy` into the allocator.
    llvm::ArrayRef<mlir::Type> element_types = allocator.copyInto(key);

    // Allocate the storage instance and construct it.
    return new (allocator.allocate<StructTypeStorage>())
        StructTypeStorage(element_types);
  }

  ArrayRef<Type> element_types_;
};
} // namespace mlir::toy::detail

//===----------------------------------------------------------------------===//
// StructType method definitions
//===----------------------------------------------------------------------===//

mlir::toy::StructType
mlir::toy::StructType::get(llvm::ArrayRef<mlir::Type> element_types) {
  assert(!element_types.empty() && "expected at least 1 element type");
  return Base::get(element_types.front().getContext(), element_types);
}

llvm::ArrayRef<mlir::Type> mlir::toy::StructType::getElementTypes() const {
  return getImpl()->element_types_;
}

::toy::usize mlir::toy::StructType::getNumElementTypes() const {
  return getElementTypes().size();
}