#ifndef TOY_DIALECT_PASSES_HPP
#define TOY_DIALECT_PASSES_HPP

#include <memory>

// Forward declarations
namespace mlir {
class Pass;
}

namespace mlir::toy {
std::unique_ptr<Pass> createShapeInferencePass();
} // namespace mlir::toy

#endif // TOY_DIALECT_PASSES_HPP