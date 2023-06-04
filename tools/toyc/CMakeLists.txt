llvm_map_components_to_libnames(llvm_libs support)

add_executable(
  toyc
  toyc.cpp
)
target_link_libraries(
  toyc
  PRIVATE
  toy_mlir_library
  range-v3
  ${llvm_libs}
  MLIRAffineTransforms
  MLIRExecutionEngine
)