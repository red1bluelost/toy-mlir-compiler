#include "toy/dialect/dialect.hpp"

#include <mlir/IR/PatternMatch.h>

namespace {
#include "simplify.cpp.inc"
}

void mlir::toy::TransposeOp::getCanonicalizationPatterns(
    RewritePatternSet& results, MLIRContext* context
) {
  results.add<TransposeTransposeOptPattern>(context);
}

void mlir::toy::ReshapeOp::getCanonicalizationPatterns(
    RewritePatternSet& results, MLIRContext* context
) {
  results.add<
      ReshapeReshapeOptPattern,
      RedundantReshapeOptPattern,
      FoldConstantReshapeOptPattern>(context);
}