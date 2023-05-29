#include "toy/dialect/dialect.hpp"
#include "toy/dialect/passes.hpp"
#include "toy/dialect/shape_inference_interface.hpp"

#include <llvm/Support/Debug.h>
#include <mlir/Pass/Pass.h>

#define DEBUG_TYPE "shape-inference"

namespace mlir::toy {
/// Include the auto-generated definitions for the shape inference interfaces.
#include "toy/dialect/shape_interface_interface.cpp.inc"
} // namespace mlir::toy

namespace {
/// The ShapeInferencePass is a pass that performs intra-procedural
/// shape inference.
///
///    Algorithm:
///
///   1) Build a worklist containing all the operations that return a
///      dynamically shaped tensor: these are the operations that need shape
///      inference.
///   2) Iterate on the worklist:
///     a) find an operation to process: the next ready operation in the
///        worklist has all of its arguments non-generic,
///     b) if no operation is found, break out of the loop,
///     c) remove the operation from the worklist,
///     d) infer the shape of its output from the argument types.
///   3) If the worklist is empty, the algorithm succeeded.
///
struct ShapeInferencePass final
    : public mlir::PassWrapper<
          ShapeInferencePass,
          mlir::OperationPass<mlir::toy::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShapeInferencePass)

  void runOnOperation() final {
    auto f = getOperation();

    // Populate the worklist with the operations that need shape inference:
    // these are operations that return a dynamic shape.
    llvm::SmallPtrSet<mlir::Operation*, 16> op_work_list;
    f.walk([&](mlir::Operation* op) {
      if (returns_dynamic_shape(op)) op_work_list.insert(op);
    });

    // Iterate on the operations in the worklist until all operations have been
    // inferred or no change happened (fix point).
    while (!op_work_list.empty()) {
      // Find the next operation ready for inference, that is an operation
      // with all operands already resolved (non-generic).
      auto nextop = llvm::find_if(op_work_list, all_operands_returned);
      if (nextop == op_work_list.end()) break;

      mlir::Operation* op = *nextop;
      op_work_list.erase(op);

      // Ask the operation to infer its output shapes.
      LLVM_DEBUG(llvm::dbgs() << "Inferring shape for: " << *op << "\n");
      if (auto shapeOp = dyn_cast<mlir::toy::ShapeInference>(op)) {
        shapeOp.inferShapes();
      } else {
        op->emitError(
            "unable to infer shape of operation without shape "
            "inference interface"
        );
        return signalPassFailure();
      }
    }

    // If the operation worklist isn't empty, this indicates a failure.
    if (!op_work_list.empty()) {
      f.emitError("Shape inference failed, ")
          << op_work_list.size() << " operations couldn't be inferred\n";
      signalPassFailure();
    }
  }

  /// A utility method that returns if the given operation has all of its
  /// operands inferred.
  static bool all_operands_returned(mlir::Operation* op) {
    return llvm::all_of(op->getOperandTypes(), [](mlir::Type operandType) {
      return llvm::isa<mlir::RankedTensorType>(operandType);
    });
  }

  /// A utility method that returns if the given operation has a dynamically
  /// shaped result.
  static bool returns_dynamic_shape(mlir::Operation* op) {
    return llvm::any_of(op->getResultTypes(), [](mlir::Type resultType) {
      return !llvm::isa<mlir::RankedTensorType>(resultType);
    });
  }
};
} // namespace

/// Create a Shape Inference pass.
std::unique_ptr<mlir::Pass> mlir::toy::createShapeInferencePass() {
  return std::make_unique<ShapeInferencePass>();
}
