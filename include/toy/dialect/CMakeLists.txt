include(TableGen)
include(AddMLIR)

set(LLVM_TARGET_DEFINITIONS ops.td)
mlir_tablegen(ops.hpp.inc -gen-op-decls)
mlir_tablegen(ops.cpp.inc -gen-op-defs)
mlir_tablegen(dialect.hpp.inc -gen-dialect-decls)
mlir_tablegen(dialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(toy_ops_inc_gen)

set(LLVM_TARGET_DEFINITIONS shape_inference_interface.td)
mlir_tablegen(shape_interface_interface.hpp.inc -gen-op-interface-decls)
mlir_tablegen(shape_interface_interface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(toy_shape_interface_interface_inc_gen)
