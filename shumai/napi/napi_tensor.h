#pragma once

#include <napi.h>
#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/autograd/tensor/AutogradExtension.h"
#include "flashlight/fl/autograd/tensor/AutogradOps.h"
#include "flashlight/fl/common/DynamicBenchmark.h"
#include "flashlight/fl/nn/Init.h"
#include "flashlight/fl/runtime/Device.h"
#include "flashlight/fl/runtime/Stream.h"
#include "flashlight/fl/tensor/Compute.h"
#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/tensor/TensorAdapter.h"

class Tensor : public Napi::ObjectWrap<Tensor> {
 public:
  Tensor(const Napi::CallbackInfo&);
  static Napi::FunctionReference* constructor;
  fl::Tensor* _tensor;

  Napi::Value Elements(const Napi::CallbackInfo&);
  Napi::Value NDim(const Napi::CallbackInfo&);
  Napi::Value Dtype(const Napi::CallbackInfo&);
  Napi::Value Bytes(const Napi::CallbackInfo&);
  Napi::Value Shape(const Napi::CallbackInfo&);
  Napi::Value Shape64(const Napi::CallbackInfo&);
  Napi::Value ToString(const Napi::CallbackInfo&);
  Napi::Value ToFloat32Array(const Napi::CallbackInfo&);
  Napi::Value ToFloat64Array(const Napi::CallbackInfo&);
  Napi::Value ToBoolInt8Array(const Napi::CallbackInfo&);
  Napi::Value ToInt16Array(const Napi::CallbackInfo&);
  Napi::Value ToInt32Array(const Napi::CallbackInfo&);
  Napi::Value ToBigInt64Array(const Napi::CallbackInfo&);
  Napi::Value ToUint8Array(const Napi::CallbackInfo&);
  Napi::Value ToUint16Array(const Napi::CallbackInfo&);
  Napi::Value ToUint32Array(const Napi::CallbackInfo&);
  Napi::Value ToBigUint64Array(const Napi::CallbackInfo&);
  Napi::Value Reshape(const Napi::CallbackInfo& info);
  Napi::Value AsType(const Napi::CallbackInfo& info);
  Napi::Value Transpose(const Napi::CallbackInfo& info);
  Napi::Value Tile(const Napi::CallbackInfo& info);
  Napi::Value NonZero(const Napi::CallbackInfo& info);
  Napi::Value Negative(const Napi::CallbackInfo& info);
  Napi::Value LogicalNot(const Napi::CallbackInfo& info);
  Napi::Value Exp(const Napi::CallbackInfo& info);
  Napi::Value Log(const Napi::CallbackInfo& info);
  Napi::Value Log1p(const Napi::CallbackInfo& info);
  Napi::Value Sin(const Napi::CallbackInfo& info);
  Napi::Value Cos(const Napi::CallbackInfo& info);
  Napi::Value Sqrt(const Napi::CallbackInfo& info);
  Napi::Value Tanh(const Napi::CallbackInfo& info);
  Napi::Value Floor(const Napi::CallbackInfo& info);
  Napi::Value Ceil(const Napi::CallbackInfo& info);
  Napi::Value Rint(const Napi::CallbackInfo& info);
  Napi::Value Absolute(const Napi::CallbackInfo& info);
  Napi::Value Sigmoid(const Napi::CallbackInfo& info);
  Napi::Value Erf(const Napi::CallbackInfo& info);
  Napi::Value Flip(const Napi::CallbackInfo& info);
  Napi::Value Clip(const Napi::CallbackInfo& info);
  Napi::Value Roll(const Napi::CallbackInfo& info);
  Napi::Value IsNaN(const Napi::CallbackInfo& info);
  Napi::Value IsInf(const Napi::CallbackInfo& info);
  Napi::Value Sign(const Napi::CallbackInfo& info);
  Napi::Value Tril(const Napi::CallbackInfo& info);
  Napi::Value Triu(const Napi::CallbackInfo& info);
  Napi::Value Where(const Napi::CallbackInfo& info);
  Napi::Value Sort(const Napi::CallbackInfo& info);
  Napi::Value Add(const Napi::CallbackInfo& info);
  Napi::Value Sub(const Napi::CallbackInfo& info);
  Napi::Value Mul(const Napi::CallbackInfo& info);
  Napi::Value Div(const Napi::CallbackInfo& info);
  Napi::Value Eq(const Napi::CallbackInfo& info);
  Napi::Value Neq(const Napi::CallbackInfo& info);
  Napi::Value LessThan(const Napi::CallbackInfo& info);
  Napi::Value LessThanEqual(const Napi::CallbackInfo& info);
  Napi::Value GreaterThan(const Napi::CallbackInfo& info);
  Napi::Value GreaterThanEqual(const Napi::CallbackInfo& info);
  Napi::Value LogicalOr(const Napi::CallbackInfo& info);
  Napi::Value LogicalAnd(const Napi::CallbackInfo& info);
  Napi::Value Mod(const Napi::CallbackInfo& info);
  Napi::Value BitwiseAnd(const Napi::CallbackInfo& info);
  Napi::Value BitwiseOr(const Napi::CallbackInfo& info);
  Napi::Value BitwiseXor(const Napi::CallbackInfo& info);
  Napi::Value LShift(const Napi::CallbackInfo& info);
  Napi::Value RShift(const Napi::CallbackInfo& info);
  Napi::Value Minimum(const Napi::CallbackInfo& info);
  Napi::Value Maximum(const Napi::CallbackInfo& info);
  Napi::Value Power(const Napi::CallbackInfo& info);
  Napi::Value MatMul(const Napi::CallbackInfo& info);
  Napi::Value Conv2d(const Napi::CallbackInfo& info);
  Napi::Value AMin(const Napi::CallbackInfo& info);
  Napi::Value AMax(const Napi::CallbackInfo& info);
  Napi::Value ArgMin(const Napi::CallbackInfo& info);
  Napi::Value ArgMax(const Napi::CallbackInfo& info);
  Napi::Value Sum(const Napi::CallbackInfo& info);
  Napi::Value CumSum(const Napi::CallbackInfo& info);
  Napi::Value Mean(const Napi::CallbackInfo& info);
  Napi::Value Median(const Napi::CallbackInfo& info);
  Napi::Value Var(const Napi::CallbackInfo& info);
  Napi::Value Std(const Napi::CallbackInfo& info);
  Napi::Value Norm(const Napi::CallbackInfo& info);
  Napi::Value CountNonZero(const Napi::CallbackInfo& info);
  Napi::Value Any(const Napi::CallbackInfo& info);
  Napi::Value All(const Napi::CallbackInfo& info);

  void Dispose(const Napi::CallbackInfo&);
  void Update(const Napi::CallbackInfo&);
  void Eval(const Napi::CallbackInfo& info);
  void Save(const Napi::CallbackInfo& info);

  static Napi::Function GetClass(Napi::Env);

 private:
  Napi::TypedArray _underlying;
  Napi::Array _deps;
};
