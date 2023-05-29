#include "toy/dialect/dialect.hpp"
#include "toy/dialect/passes.hpp"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/TypeID.h>
#include <mlir/Transforms/DialectConversion.h>

#include <concepts>
#include <memory>
#include <ranges>

#define DEBUG_TYPE "toy-lower-to-affine"

namespace {
//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns
//===----------------------------------------------------------------------===//

mlir::MemRefType convert_tensor_to_memref(mlir::RankedTensorType type) {
  return mlir::MemRefType::get(type.getShape(), type.getElementType());
}

static mlir::Value insert_alloc_dealloc(
    mlir::MemRefType type, mlir::Location loc, mlir::PatternRewriter& rewriter
) {
  auto alloc = rewriter.create<mlir::memref::AllocOp>(loc, type);

  // Make sure to allocate at the beginning of the block.
  mlir::Block* parent_block = alloc->getBlock();
  alloc->moveBefore(&parent_block->front());

  // Make sure to deallocate this alloc at the end of the block. This is fine
  // as toy functions have no control flow.
  auto dealloc = rewriter.create<mlir::memref::DeallocOp>(loc, alloc);
  dealloc->moveBefore(&parent_block->back());
  return alloc;
}

/// This defines the function type used to process an iteration of a lowered
/// loop. It takes as input an OpBuilder, an range of memRefOperands
/// corresponding to the operands of the input operation, and the range of loop
/// induction variables for the iteration. It returns a value to store at the
/// current index of the iteration.
template<typename Func>
concept LoopIterationInvocable =
    std::invocable<Func, mlir::OpBuilder&, mlir::ValueRange, mlir::ValueRange>;

void lower_op_to_loops(
    mlir::Operation*            op,
    mlir::ValueRange            operands,
    mlir::PatternRewriter&      rewriter,
    LoopIterationInvocable auto process_iteration
) {
  auto tensor_type =
      llvm::cast<mlir::RankedTensorType>(op->getResultTypes().front());
  mlir::Location loc = op->getLoc();

  mlir::MemRefType memref_type = convert_tensor_to_memref(tensor_type);
  mlir::Value      alloc = insert_alloc_dealloc(memref_type, loc, rewriter);

  llvm::SmallVector<toy::i64, 4> lower_bounds(
      static_cast<size_t>(tensor_type.getRank()), /*Value=*/0
  );
  llvm::SmallVector<toy::i64, 4> steps(
      static_cast<size_t>(tensor_type.getRank()), /*Value=*/1
  );
  mlir::affine::buildAffineLoopNest(
      rewriter,
      loc,
      lower_bounds,
      tensor_type.getShape(),
      steps,
      [&](mlir::OpBuilder& nested_builder,
          mlir::Location   loc,
          mlir::ValueRange ivs) {
        nested_builder.create<mlir::affine::AffineStoreOp>(
            loc, process_iteration(nested_builder, operands, ivs), alloc, ivs
        );
      }
  );

  rewriter.replaceOp(op, alloc);
}

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Binary operations
//===----------------------------------------------------------------------===//

template<typename BinaryOp, typename LoweredBinaryOp>
struct BinaryOpLowering final : mlir::ConversionPattern {
  /*implicit*/ BinaryOpLowering(mlir::MLIRContext* ctx)
      : ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::Operation*                 op,
      llvm::ArrayRef<mlir::Value>      operands,
      mlir::ConversionPatternRewriter& rewriter
  ) const final {
    mlir::Location loc = op->getLoc();
    lower_op_to_loops(
        op,
        operands,
        rewriter,
        [loc](
            mlir::OpBuilder& builder,
            mlir::ValueRange mem_ref_operands,
            mlir::ValueRange loop_ivs
        ) {
          typename BinaryOp::Adaptor binaryAdaptor = mem_ref_operands;
          return builder.create<LoweredBinaryOp>(
              loc,
              builder.create<mlir::affine::AffineLoadOp>(
                  loc, binaryAdaptor.getLhs(), loop_ivs
              ),
              builder.create<mlir::affine::AffineLoadOp>(
                  loc, binaryAdaptor.getRhs(), loop_ivs
              )
          );
        }
    );
    return mlir::success();
  }
};

using AddOpLowering = BinaryOpLowering<mlir::toy::AddOp, mlir::arith::AddFOp>;
using MulOpLowering = BinaryOpLowering<mlir::toy::MulOp, mlir::arith::MulFOp>;

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Constant operations
//===----------------------------------------------------------------------===//

struct ConstantOpLowering final
    : mlir::OpRewritePattern<mlir::toy::ConstantOp> {
  using OpRewritePattern<mlir::toy::ConstantOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::toy::ConstantOp op, mlir::PatternRewriter& rewriter
  ) const final {
    mlir::DenseElementsAttr const_val = op.getValue();
    mlir::Location          loc       = op.getLoc();

    auto tensor_type = llvm::cast<mlir::RankedTensorType>(op.getType());
    mlir::MemRefType memref_type = convert_tensor_to_memref(tensor_type);
    mlir::Value      alloc = insert_alloc_dealloc(memref_type, loc, rewriter);

    auto value_shape = memref_type.getShape();

    llvm::SmallVector<mlir::Value, 8> constant_indices;
    if (value_shape.empty()) {
      constant_indices.push_back(
          rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0)
      );
    } else {
      auto ops =
          std::views::iota(toy::i64{0}, *std::ranges::max_element(value_shape))
          | std::views::transform([&](toy::i64 idx) {
              return rewriter.create<mlir::arith::ConstantIndexOp>(loc, idx);
            });
      constant_indices.append(ops.begin(), ops.end());
    }

    llvm::SmallVector<mlir::Value, 2> indices;
    auto value_it = const_val.value_begin<mlir::FloatAttr>();
    std::function<void(toy::u64)> store_elms = [&](toy::u64 dimension) {
      if (dimension == value_shape.size()) {
        rewriter.create<mlir::affine::AffineStoreOp>(
            loc,
            rewriter.create<mlir::arith::ConstantOp>(loc, *value_it++),
            alloc,
            indices
        );
        return;
      }

      for (toy::i64 idx :
           std::views::iota(toy::i64{0}, value_shape[dimension])) {
        indices.push_back(constant_indices[static_cast<size_t>(idx)]);
        store_elms(dimension + 1);
        indices.pop_back();
      }
    };

    store_elms(0);

    rewriter.replaceOp(op, alloc);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Func operations
//===----------------------------------------------------------------------===//

struct FuncOpLowering final : mlir::OpConversionPattern<mlir::toy::FuncOp> {
  using OpConversionPattern<mlir::toy::FuncOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::toy::FuncOp op, OpAdaptor, mlir::ConversionPatternRewriter& rewriter
  ) const final {
    if (op.getName() != "main") {
      return rewriter.notifyMatchFailure(op, [](auto& diag) {
        diag << "expected non-main functions to be inlined";
      });
    }
    if (op.getNumArguments() || op.getFunctionType().getNumInputs()) {
      return rewriter.notifyMatchFailure(op, [](auto& diag) {
        diag << "expected 'main' to have 0 inputs and 0 results";
      });
    }

    auto func = rewriter.create<mlir::func::FuncOp>(
        op.getLoc(), op.getName(), op.getFunctionType()
    );
    rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Print operations
//===----------------------------------------------------------------------===//

struct PrintOpLowering final : mlir::OpConversionPattern<mlir::toy::PrintOp> {
  using OpConversionPattern<mlir::toy::PrintOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::toy::PrintOp               op,
      OpAdaptor                        adaptor,
      mlir::ConversionPatternRewriter& rewriter
  ) const final {
    rewriter.updateRootInPlace(op, [&] {
      op->setOperands(adaptor.getOperands());
    });
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Return operations
//===----------------------------------------------------------------------===//

struct ReturnOpLowering final : mlir::OpRewritePattern<mlir::toy::ReturnOp> {
  using OpRewritePattern<mlir::toy::ReturnOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::toy::ReturnOp op, mlir::PatternRewriter& rewriter
  ) const final {
    if (op.hasOperand()) return mlir::failure();
    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Transpose operations
//===----------------------------------------------------------------------===//

struct TransposeOpLowering final : mlir::ConversionPattern {
  /*implicit*/ TransposeOpLowering(mlir::MLIRContext* ctx)
      : ConversionPattern(mlir::toy::TransposeOp::getOperationName(), 1, ctx) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::Operation*                 op,
      llvm::ArrayRef<mlir::Value>      operands,
      mlir::ConversionPatternRewriter& rewriter
  ) const final {
    mlir::Location loc = op->getLoc();
    lower_op_to_loops(
        op,
        operands,
        rewriter,
        [loc](
            mlir::OpBuilder& builder,
            mlir::ValueRange mem_ref_operands,
            mlir::ValueRange loop_ivs
        ) {
          mlir::toy::TransposeOpAdaptor adaptor = mem_ref_operands;
          mlir::Value                   input   = adaptor.getInput();

          llvm::SmallVector<mlir::Value, 2> reverse_ivs{
              llvm::reverse(loop_ivs)};
          return builder.create<mlir::affine::AffineLoadOp>(
              loc, input, reverse_ivs
          );
        }
    );
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffineLoweringPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to affine loops of the toy operations that are
/// computationally intensive (like matmul for example...) while keeping the
/// rest of the code in the Toy dialect.
struct ToyToAffineLoweringPass final
    : mlir::PassWrapper<
          ToyToAffineLoweringPass,
          mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyToAffineLoweringPass)

  void getDependentDialects(mlir::DialectRegistry& registry) const final {
    registry.insert<
        mlir::affine::AffineDialect,
        mlir::func::FuncDialect,
        mlir::memref::MemRefDialect>();
  }

  void runOnOperation() final {
    mlir::ConversionTarget target(getContext());

    target.addLegalDialect<
        mlir::BuiltinDialect,
        mlir::affine::AffineDialect,
        mlir::arith::ArithDialect,
        mlir::func::FuncDialect,
        mlir::memref::MemRefDialect>();

    target.addIllegalDialect<mlir::toy::ToyDialect>();
    target.addDynamicallyLegalOp<mlir::toy::PrintOp>([](mlir::toy::PrintOp op) {
      return llvm::none_of(op->getOperandTypes(), [](mlir::Type type) {
        return llvm::isa<mlir::TensorType>(type);
      });
    });

    mlir::MLIRContext*      ctx      = &getContext();
    mlir::RewritePatternSet patterns = ctx;
    patterns.add<
        AddOpLowering,
        ConstantOpLowering,
        FuncOpLowering,
        MulOpLowering,
        PrintOpLowering,
        ReturnOpLowering,
        TransposeOpLowering>(ctx);

    if (failed(
            applyPartialConversion(getOperation(), target, std::move(patterns))
        ))
      signalPassFailure();
  }
};
} // namespace

/// Create a pass for lowering operations in the `Affine` and `Std` dialects,
/// for a subset of the Toy IR (e.g. matmul).
std::unique_ptr<mlir::Pass> mlir::toy::createLowerToAffinePass() {
  return std::make_unique<ToyToAffineLoweringPass>();
}