// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

// Some of the tests can't run on TensorrtExecutionProvider because axis=0 is not supported
// or there are unsupported data types. Those tests will fallback to other EPs.

TEST(SqueezeOpTest, Squeeze_1) {
  OpTester test("Squeeze");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddInput<float>("data", {1, 3, 4, 5}, std::vector<float>(60, 1.0f));
  test.AddOutput<float>("squeezed", {3, 4, 5}, std::vector<float>(60, 1.0f));
  test.Run();
}

TEST(SqueezeOpTest, Squeeze_Empty_Axes_1) {
  OpTester test("Squeeze");
  test.AddInput<float>("data", {1, 1, 4, 1}, std::vector<float>(4, 1.0f));
  test.AddOutput<float>("squeezed", {4}, std::vector<float>(4, 1.0f));
  // TensorRT doesn't seem to support missing 'axes'
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",  {kTensorrtExecutionProvider});
}

TEST(SqueezeOpTest, Squeeze_Empty_Axes_2) {
  OpTester test("Squeeze");
  // nothing to "squeeze" out in the input shape
  test.AddInput<float>("data", {2, 4}, std::vector<float>(8, 1.0f));
  test.AddOutput<float>("squeezed", {2, 4}, std::vector<float>(8, 1.0f));
  // TensorRT doesn't seem to support missing 'axes'
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(SqueezeOpTest, Squeeze_1_int32) {
  OpTester test("Squeeze");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddInput<int32_t>("data", {1, 3, 4, 5}, std::vector<int32_t>(60, 1));
  test.AddOutput<int32_t>("squeezed", {3, 4, 5}, std::vector<int32_t>(60, 1));
  test.Run();
}

TEST(SqueezeOpTest, Squeeze_string) {
  OpTester test("Squeeze");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2, 4});
  test.AddInput<std::string>("data", {1, 2, 1, 3, 1}, std::vector<std::string>({"1", "2", "3", "4", "5", "6"}));
  test.AddOutput<std::string>("squeezed", {2, 3}, std::vector<std::string>({"1", "2", "3", "4", "5", "6"}));
  test.Run();
}

TEST(SqueezeOpTest, Squeeze_2) {
  OpTester test("Squeeze");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2, 3});
  test.AddInput<float>("data", {1, 4, 1, 1, 2},
                       std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  test.AddOutput<float>("squeezed", {4, 2},
                        std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  test.Run();
}

TEST(SqueezeOpTest, UnsortedAxes) {
  OpTester test("Squeeze");
  test.AddShapeToTensorData(false);  // TODO: re-enable shape inference test after ONNX fix
  test.AddAttribute("axes", std::vector<int64_t>{3, 0, 2});
  test.AddInput<float>("data", {1, 4, 1, 1, 2},
                       std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  test.AddOutput<float>("squeezed", {4, 2},
                        std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  test.Run();
}

TEST(SqueezeOpTest, DuplicateAxes) {
  OpTester test("Squeeze");
  test.AddShapeToTensorData(false);  // TODO: re-enable shape inference test after ONNX fix
  test.AddAttribute("axes", std::vector<int64_t>{3, 0, 2, 0, 2, 3});
  test.AddInput<float>("data", {1, 4, 1, 1, 2},
                       std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  test.AddOutput<float>("squeezed", {4, 2},
                        std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  test.Run();
}

TEST(SqueezeOpTest, BadAxes) {
  OpTester test("Squeeze");
  test.AddShapeToTensorData(false);  // TODO: re-enable shape inference test after ONNX fix
  // Bad axes - should be 1 instead of 0.
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddInput<float>("data", {3, 1, 4, 5}, std::vector<float>(60, 1.0f));
  test.AddOutput<float>("squeezed", {3, 4, 5}, std::vector<float>(60, 1.0f));

  // Expect failure.
  test.Run(OpTester::ExpectResult::kExpectFailure, "Dimension of input 0 must be 1 instead of 3", {kTensorrtExecutionProvider});
}

TEST(SqueezeOpTest, SqueezeNegAxis_2) {
  OpTester test("Squeeze", 11);
  test.AddAttribute("axes", std::vector<int64_t>{0, -3, -2});
  test.AddInput<float>("data", {1, 4, 1, 1, 2},
                       std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  test.AddOutput<float>("squeezed", {4, 2},
                        std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  
  // nGraph does not support neg axis.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",  {kNGraphExecutionProvider});
}

}  // namespace test
}  // namespace onnxruntime
