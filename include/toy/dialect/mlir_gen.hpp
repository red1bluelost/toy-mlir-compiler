#ifndef TOY_DIALECT_MLIR_GEN_HPP
#define TOY_DIALECT_MLIR_GEN_HPP

// Forward declarations
namespace mlir {
class MLIRContext;
class ModuleOp;
template<typename>
class OwningOpRef;
} // namespace mlir

// Forward declarations
namespace toy {
class ModuleAST;
}

namespace toy {
/// Emit IR for the given Toy moduleAST, returns a newly created MLIR module
/// or nullptr on failure.
mlir::OwningOpRef<mlir::ModuleOp>
mlir_gen(mlir::MLIRContext& context, const ModuleAST& module_ast);
} // namespace toy

#endif // TOY_DIALECT_MLIR_GEN_HPP
