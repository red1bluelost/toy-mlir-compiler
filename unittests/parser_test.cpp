#include "toy/parser/parser.hpp"

#include "toy/util/string_view_line_reader.hpp"

#include <llvm/Support/Casting.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace {

using ::testing::IsEmpty, ::testing::Eq, ::testing::ElementsAre,
    ::testing::Truly, ::testing::SizeIs, ::testing::Matcher,
    ::testing::MatchResultListener, ::testing::PolymorphicMatcher,
    ::testing::MakePolymorphicMatcher, ::testing::Property, ::testing::NotNull,
    ::testing::Field;

template<typename To>
class WhenDynCastToMatcher {
 public:
  explicit WhenDynCastToMatcher(const Matcher<To>& matcher)
      : matcher_(matcher) {}

  void DescribeTo(::std::ostream* os) const { matcher_.DescribeTo(os); }

  void DescribeNegationTo(::std::ostream* os) const {
    matcher_.DescribeNegationTo(os);
  }

  template<typename From>
  bool MatchAndExplain(From* from, MatchResultListener* listener) const {
    if (!llvm::isa<To>(from)) return false;
    return MatchPrintAndExplain(
        llvm::cast<To>(*from), this->matcher_, listener
    );
  }

  template<typename From>
  bool MatchAndExplain(From& from, MatchResultListener* listener) const {
    if (!llvm::isa<To>(from)) return false;
    return MatchPrintAndExplain(llvm::cast<To>(from), this->matcher_, listener);
  }

  template<typename From>
  bool MatchAndExplain(
      const std::unique_ptr<From>& from, MatchResultListener* listener
  ) const {
    if (!llvm::isa<To>(from)) return false;
    return MatchPrintAndExplain(
        llvm::cast<To>(*from), this->matcher_, listener
    );
  }

 private:
  const Matcher<To> matcher_;
};

template<typename To>
PolymorphicMatcher<WhenDynCastToMatcher<To>>
WhenDynCastTo(const Matcher<To>& inner_matcher) {
  return MakePolymorphicMatcher(WhenDynCastToMatcher<To>(inner_matcher));
}

auto IsNumberExpr(
    const Matcher<toy::f64>&      value_match = testing::_,
    const Matcher<toy::Location>& loc_match   = testing::_
) {
  return WhenDynCastTo<toy::NumberExprAST>(testing::AllOf(
      Property(&toy::NumberExprAST::value, value_match),
      Property(&toy::ExprAST::location, loc_match)
  ));
}

auto IsLiteralExpr(
    const Matcher<std::span<const toy::i64>>& dim_match = testing::_,
    const Matcher<std::span<const std::unique_ptr<toy::ExprAST>>>&
                                  values_match = testing::_,
    const Matcher<toy::Location>& loc_match    = testing::_

) {
  return WhenDynCastTo<toy::LiteralExprAST>(testing::AllOf(
      Property(&toy::LiteralExprAST::dims, dim_match),
      Property(&toy::LiteralExprAST::values, values_match),
      Property(&toy::ExprAST::location, loc_match)
  ));
}

auto IsVariableExpr(
    const Matcher<std::string_view>& name_match = testing::_,
    const Matcher<toy::Location>&    loc_match  = testing::_
) {
  return WhenDynCastTo<toy::VariableExprAST>(testing::AllOf(
      Property(&toy::VariableExprAST::name, name_match),
      Property(&toy::ExprAST::location, loc_match)
  ));
}

auto IsVarDeclExpr(
    const Matcher<std::string_view>& name_match     = testing::_,
    const Matcher<toy::ExprAST>&     init_val_match = testing::_,
    const Matcher<toy::VarType>&     type_match     = testing::_,
    const Matcher<toy::Location>&    loc_match      = testing::_
) {
  return WhenDynCastTo<toy::VarDeclExprAST>(testing::AllOf(
      Property(&toy::VarDeclExprAST::name, name_match),
      Property(&toy::VarDeclExprAST::init_val, init_val_match),
      Property(&toy::VarDeclExprAST::type, type_match),
      Property(&toy::ExprAST::location, loc_match)
  ));
}

auto IsBinaryExpr(
    const Matcher<toy::u8>&       op_match  = testing::_,
    const Matcher<toy::ExprAST>&  lhs_match = testing::_,
    const Matcher<toy::ExprAST>&  rhs_match = testing::_,
    const Matcher<toy::Location>& loc_match = testing::_
) {
  return WhenDynCastTo<toy::BinaryExprAST>(testing::AllOf(
      Property(&toy::BinaryExprAST::op, op_match),
      Property(&toy::BinaryExprAST::lhs, lhs_match),
      Property(&toy::BinaryExprAST::rhs, rhs_match),
      Property(&toy::ExprAST::location, loc_match)
  ));
}

auto IsCallExpr(
    const Matcher<std::string_view>& callee_match = testing::_,
    const Matcher<std::span<const std::unique_ptr<toy::ExprAST>>>& args_match =
        testing::_,
    const Matcher<toy::Location>& loc_match = testing::_
) {
  return WhenDynCastTo<toy::CallExprAST>(testing::AllOf(
      Property(&toy::CallExprAST::callee, callee_match),
      Property(&toy::CallExprAST::args, args_match),
      Property(&toy::ExprAST::location, loc_match)
  ));
}

auto IsPrintExpr(
    const Matcher<toy::ExprAST>&  arg_match = testing::_,
    const Matcher<toy::Location>& loc_match = testing::_
) {
  return WhenDynCastTo<toy::PrintExprAST>(testing::AllOf(
      Property(&toy::PrintExprAST::arg, arg_match),
      Property(&toy::ExprAST::location, loc_match)
  ));
}

auto IsPrototypeAST(
    const Matcher<std::string_view>& name_match = testing::_,
    const Matcher<std::span<const std::unique_ptr<toy::VariableExprAST>>>&
                                  args_match = testing::_,
    const Matcher<toy::Location>& loc_match  = testing::_
) {
  return testing::AllOf(
      Property(&toy::PrototypeAST::name, name_match),
      Property(&toy::PrototypeAST::args, args_match),
      Property(&toy::PrototypeAST::location, loc_match)
  );
}

auto IsFunctionAST(
    const Matcher<toy::PrototypeAST>& proto_match = testing::_,
    const Matcher<std::span<const std::unique_ptr<toy::ExprAST>>>& body_match =
        testing::_
) {
  return testing::AllOf(
      Property(&toy::FunctionAST::proto, proto_match),
      Property(&toy::FunctionAST::body, body_match)
  );
}

auto IsModuleAST(
    const Matcher<std::span<const toy::FunctionAST>>& functions_match =
        testing::_
) {
  return testing::AllOf(Property(&toy::ModuleAST::functions, functions_match));
}

auto IsVarType(const Matcher<llvm::SmallVector<toy::i64, 2>>& shape_match) {
  return testing::AllOf(Field(&toy::VarType::shape, shape_match));
}

template<std::same_as<toy::f64>... Floats>
auto Is1DLiteral(Floats... floats) {
  return IsLiteralExpr(
      ElementsAre(sizeof...(Floats)), ElementsAre(IsNumberExpr(floats)...)
  );
}

TEST(Parser, EmptyMain) {
  // Arrange
  std::string_view test_name =
      ::testing::UnitTest::GetInstance()->current_test_info()->name();
  toy::Parser parser(
      toy::Lexer(test_name, toy::StringViewLineReader("fn main() {}"))
  );

  // Act
  auto AST = parser.parse_module();

  // Assert
  ASSERT_THAT(AST, NotNull());
  ASSERT_THAT(
      *AST,
      IsModuleAST(ElementsAre(IsFunctionAST(
          IsPrototypeAST("main", IsEmpty(), toy::Location{test_name, 1, 1}),
          IsEmpty()
      )))
  );
}

TEST(Parser, MainWithPrintLiteral) {
  // Arrange
  std::string_view test_name =
      ::testing::UnitTest::GetInstance()->current_test_info()->name();
  toy::Parser parser(toy::Lexer(test_name, toy::StringViewLineReader(R"(
fn main() {
  print([1]);
}
)")));

  // Act
  auto AST = parser.parse_module();

  // Assert
  ASSERT_THAT(AST, NotNull());
  ASSERT_THAT(
      *AST,
      IsModuleAST(ElementsAre(IsFunctionAST(
          IsPrototypeAST("main", IsEmpty(), toy::Location{test_name, 2, 1}),
          ElementsAre(IsPrintExpr(
              IsLiteralExpr(
                  ElementsAre(1),
                  ElementsAre(IsNumberExpr(1.0, toy::Location{test_name, 3, 10})
                  ),
                  toy::Location{test_name, 3, 9}
              ),
              toy::Location{test_name, 3, 3}
          ))
      )))
  );
}

TEST(Parser, MainChapter1Ex1) {
  // Arrange
  std::string_view test_name =
      ::testing::UnitTest::GetInstance()->current_test_info()->name();
  toy::Parser parser(toy::Lexer(test_name, toy::StringViewLineReader(R"(
fn main() {
  # Define a variable `a` with shape <2, 3>, initialized with the literal value.
  # The shape is inferred from the supplied literal.
  let a = [[1, 2, 3], [4, 5, 6]];

  # b is identical to a, the literal tensor is implicitly reshaped: defining new
  # variables is the way to reshape tensors (element count must match).
  let b: <2, 3> = [1, 2, 3, 4, 5, 6];

  # transpose() and print() are the only builtin, the following will transpose
  # a and b and perform an element-wise multiplication before printing the result.
  print(transpose(a) * transpose(b));
}
)")));

  // Act
  auto AST = parser.parse_module();

  // Assert
  ASSERT_THAT(AST, NotNull());
  ASSERT_THAT(
      *AST,
      IsModuleAST(ElementsAre(IsFunctionAST(
          IsPrototypeAST("main", IsEmpty()),
          ElementsAre(
              IsVarDeclExpr(
                  "a",
                  IsLiteralExpr(
                      ElementsAre(2, 3),
                      ElementsAre(
                          Is1DLiteral(1.0, 2.0, 3.0), Is1DLiteral(4.0, 5.0, 6.0)
                      )
                  ),
                  IsVarType(IsEmpty())
              ),
              IsVarDeclExpr(
                  "b",
                  Is1DLiteral(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
                  IsVarType(ElementsAre(2, 3))
              ),
              IsPrintExpr(IsBinaryExpr(
                  '*',
                  IsCallExpr("transpose", ElementsAre(IsVariableExpr("a"))),
                  IsCallExpr("transpose", ElementsAre(IsVariableExpr("b")))
              ))
          )
      )))
  );
}

} // namespace
