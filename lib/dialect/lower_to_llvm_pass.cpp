#include "toy/dialect/dialect.hpp"
#include "toy/dialect/passes.hpp"

#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include <string_view>

namespace {
//===----------------------------------------------------------------------===//
// ToyToLLVM RewritePatterns: Print Operations
//===----------------------------------------------------------------------===//

/// Lowers `toy.print` to a loop nest calling `printf` on each of the individual
/// elements of the array.
class PrintOpLowering : public mlir::ConversionPattern {
 public:
  explicit PrintOpLowering(mlir::MLIRContext* ctx)
      : ConversionPattern(mlir::toy::PrintOp::getOperationName(), 1, ctx) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::Operation* op,
      llvm::ArrayRef<mlir::Value>,
      mlir::ConversionPatternRewriter& rewriter
  ) const final {
    auto memref_type   = cast<mlir::MemRefType>(op->getOperandTypes().front());
    auto memref_shape  = memref_type.getShape();
    mlir::Location loc = op->getLoc();

    auto parent_mod = op->getParentOfType<mlir::ModuleOp>();

    mlir::FlatSymbolRefAttr printf_ref =
        get_or_insert_printf(rewriter, parent_mod);
    mlir::Value fmt_specifier_constant = get_or_create_global_string(
        loc, rewriter, "frmt_spec", llvm::StringRef("%f \0", 4), parent_mod
    );
    mlir::Value new_line_constant = get_or_create_global_string(
        loc, rewriter, "nl", mlir::StringRef("\n\0", 2), parent_mod
    );

    llvm::SmallVector<mlir::Value, 4> loop_ivs;
    size_t                            memref_size = memref_shape.size();
    loop_ivs.reserve(memref_size);
    for (auto [idx, memref_dim] : llvm::enumerate(memref_shape)) {
      auto lower_bound = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
      auto upper_bound =
          rewriter.create<mlir::arith::ConstantIndexOp>(loc, memref_dim);
      auto step = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
      auto loop = rewriter.create<mlir::scf::ForOp>(
          loc, lower_bound, upper_bound, step
      );
      for (mlir::Operation& nested : *loop.getBody()) rewriter.eraseOp(&nested);
      loop_ivs.push_back(loop.getInductionVar());

      rewriter.setInsertionPointToEnd(loop.getBody());

      if (idx != memref_size - 1) {
        rewriter.create<mlir::func::CallOp>(
            loc, printf_ref, rewriter.getIntegerType(32), new_line_constant
        );
      }
      rewriter.create<mlir::scf::YieldOp>(loc);
      rewriter.setInsertionPointToStart(loop.getBody());
    }

    auto print_op     = cast<mlir::toy::PrintOp>(op);
    auto element_load = rewriter.create<mlir::memref::LoadOp>(
        loc, print_op.getInput(), loop_ivs
    );
    rewriter.create<mlir::func::CallOp>(
        loc,
        printf_ref,
        rewriter.getIntegerType(32),
        llvm::ArrayRef<mlir::Value>{fmt_specifier_constant, element_load}
    );

    rewriter.eraseOp(op);
    return mlir::success();
  }

 private:
  static mlir::FlatSymbolRefAttr
  get_or_insert_printf(mlir::PatternRewriter& rewriter, mlir::ModuleOp mod) {
    constexpr std::string_view printf_name = "printf";
    mlir::MLIRContext*         ctx         = mod.getContext();
    if (mod.lookupSymbol<mlir::LLVM::LLVMFuncOp>(printf_name))
      return mlir::SymbolRefAttr::get(ctx, printf_name);

    auto llvm_fn_type = mlir::LLVM::LLVMFunctionType::get(
        mlir::IntegerType::get(ctx, 32),
        mlir::LLVM::LLVMPointerType::get(mlir::IntegerType::get(ctx, 8)),
        /*isVarArg=*/true
    );

    mlir::PatternRewriter::InsertionGuard insert_guard{rewriter};
    rewriter.setInsertionPointToStart(mod.getBody());
    rewriter.create<mlir::LLVM::LLVMFuncOp>(
        mod.getLoc(), printf_name, llvm_fn_type
    );

    return mlir::SymbolRefAttr::get(ctx, printf_name);
  }

  static mlir::Value get_or_create_global_string(
      mlir::Location   loc,
      mlir::OpBuilder& builder,
      llvm::StringRef  name,
      llvm::StringRef  value,
      mlir::ModuleOp   mod
  ) {
    auto global = mod.lookupSymbol<mlir::LLVM::GlobalOp>(name);
    if (!global) {
      mlir::OpBuilder::InsertionGuard insert_guard{builder};
      builder.setInsertionPointToStart(mod.getBody());
      global = builder.create<mlir::LLVM::GlobalOp>(
          loc,
          mlir::LLVM::LLVMArrayType::get(
              mlir::IntegerType::get(builder.getContext(), 8),
              static_cast<unsigned>(value.size())
          ),
          /*isConstant=*/true,
          mlir::LLVM::Linkage::Internal,
          name,
          builder.getStringAttr(value),
          /*alignment=*/0
      );
    }

    mlir::Value constant = builder.create<mlir::LLVM::ConstantOp>(
        loc, builder.getI64Type(), builder.getIndexAttr(0)
    );
    return builder.create<mlir::LLVM::GEPOp>(
        loc,
        mlir::LLVM::LLVMPointerType::get(
            mlir::IntegerType::get(builder.getContext(), 8)
        ),
        builder.create<mlir::LLVM::AddressOfOp>(loc, global),
        llvm::ArrayRef{constant, constant}
    );
  }
};

struct ToyToLLVMLoweringPass final
    : public mlir::PassWrapper<
          ToyToLLVMLoweringPass,
          mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyToLLVMLoweringPass)

  void getDependentDialects(mlir::DialectRegistry& registry) const override {
    registry.insert<mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect>();
  }

  void runOnOperation() final {
    mlir::MLIRContext& ctx = getContext();

    mlir::LLVMConversionTarget target{ctx};
    target.addLegalOp<mlir::ModuleOp>();

    mlir::LLVMTypeConverter type_converter{&ctx};

    mlir::RewritePatternSet patterns{&ctx};
    mlir::populateAffineToStdConversionPatterns(patterns);
    mlir::populateSCFToControlFlowConversionPatterns(patterns);
    mlir::arith::populateArithToLLVMConversionPatterns(
        type_converter, patterns
    );
    mlir::populateFinalizeMemRefToLLVMConversionPatterns(
        type_converter, patterns
    );
    mlir::cf::populateControlFlowToLLVMConversionPatterns(
        type_converter, patterns
    );
    mlir::populateFuncToLLVMConversionPatterns(type_converter, patterns);

    patterns.add<PrintOpLowering>(&ctx);

    mlir::ModuleOp mod = getOperation();
    if (failed(mlir::applyFullConversion(mod, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> mlir::toy::createLowerToLLVMPass() {
  return std::make_unique<ToyToLLVMLoweringPass>();
}