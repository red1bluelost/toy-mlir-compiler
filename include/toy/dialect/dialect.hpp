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

/// Include auto-generated dialect
#include "toy/dialect/dialect.hpp.inc"

/// Include auto-generated operations
#define GET_OP_CLASSES
#include "toy/dialect/ops.hpp.inc"

#endif // TOY_DIALECT_DIALECT_HPP