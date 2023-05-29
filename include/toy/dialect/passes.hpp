#ifndef TOY_DIALECT_PASSES_HPP
#define TOY_DIALECT_PASSES_HPP

#include <memory>

// Forward declarations
namespace mlir {
class Pass;
}

namespace mlir::toy {
/// Create a pass to resolve the unknown types in the module.
std::unique_ptr<Pass> createShapeInferencePass();

/// Create a pass for lowering to operations in the `Affine` and `Std` dialects,
/// for a subset of the Toy IR (e.g. matmul).
std::unique_ptr<mlir::Pass> createLowerToAffinePass();
} // namespace mlir::toy

#endif // TOY_DIALECT_PASSES_HPP