#ifndef TOY_DIALECT_DIALECT_HPP
#define TOY_DIALECT_DIALECT_HPP

#include "toy/core/types.hpp"
#include "toy/dialect/shape_inference_interface.hpp"

#include <llvm/ADT/ArrayRef.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/FunctionInterfaces.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/CastInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <span>

// Forward declarations
namespace mlir::toy::detail {
struct StructTypeStorage;
}

/// Include auto-generated dialect
#include "toy/dialect/dialect.hpp.inc"

/// Include auto-generated operations
#define GET_OP_CLASSES
#include "toy/dialect/ops.hpp.inc"

namespace mlir::toy {
//===----------------------------------------------------------------------===//
// Toy Types
//===----------------------------------------------------------------------===//

class StructType
    : public mlir::Type::
          TypeBase<StructType, mlir::Type, detail::StructTypeStorage> {
 public:
  /// Inherit constructors from \c TypeBase
  using Base::Base;

  /// Create an instance of a `StructType` with the given element types. There
  /// *must* be at least one element type.
  static StructType get(llvm::ArrayRef<mlir::Type> element_types);

  llvm::ArrayRef<mlir::Type> getElementTypes() const;

  ::toy::usize getNumElementTypes() const;
};
} // namespace mlir::toy

#endif // TOY_DIALECT_DIALECT_HPP