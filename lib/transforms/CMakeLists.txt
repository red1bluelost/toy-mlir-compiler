set(LLVM_TARGET_DEFINITIONS simplify.td)
mlir_tablegen(../../lib_include/simplify.cpp.inc -gen-rewriters)
add_public_tablegen_target(toy_simplify_inc_gen)
