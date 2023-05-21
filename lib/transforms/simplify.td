#ifndef TRANSFORMS_SIMPLIFY_TD
#define TRANSFORMS_SIMPLIFY_TD

include "toy/dialect/ops.td"

include "mlir/IR/PatternBase.td"

//===----------------------------------------------------------------------===//
// Basic Pattern-Match and Rewrite
//===----------------------------------------------------------------------===//

// Transpose(Transpose(x)) = Transpose(x)
def TransposeTransposeOptPattern : Pat<(TransposeOp(TransposeOp $arg)),
                                        (replaceWithValue $arg)>;

// Reshape(Reshape(x)) = Reshape(x)
def ReshapeReshapeOptPattern : Pat<(ReshapeOp(ReshapeOp $arg)),
                                   (ReshapeOp $arg)>;

//===----------------------------------------------------------------------===//
// Pattern-Match and Rewrite using Native Code Call
//===----------------------------------------------------------------------===//

def ReshapeConstant :
  NativeCodeCall<"$0.reshape(::llvm::cast<::mlir::ShapedType>($1.getType()))">;

// Reshape(Constant(x)) = x'
def FoldConstantReshapeOptPattern : Pat<
  (ReshapeOp:$res (ConstantOp $arg)),
  (ConstantOp (ReshapeConstant $arg, $res))>;

//===----------------------------------------------------------------------===//
// Pattern-Match and Rewrite with Constraints
//===----------------------------------------------------------------------===//

def TypesAreIdentical : Constraint<CPred<"$0.getType() == $1.getType()">>;

// Reshape(x) = x, where input and output shapes are identical
def RedundantReshapeOptPattern : Pat<
  (ReshapeOp:$res $arg), (replaceWithValue $arg),
  [(TypesAreIdentical $res, $arg)]>;

#endif // TRANSFORMS_SIMPLIFY_TD


