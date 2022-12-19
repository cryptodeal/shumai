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
  static Napi::FunctionReference constructor;
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
  void Dispose(const Napi::CallbackInfo&);
  void Update(const Napi::CallbackInfo&);
  void Eval(const Napi::CallbackInfo& info);
  void Save(const Napi::CallbackInfo& info);
  void Reshape(const Napi::CallbackInfo& info);

  static Napi::Function GetClass(Napi::Env);

 private:
  Napi::TypedArray _underlying;
  Napi::Array _deps;
};
