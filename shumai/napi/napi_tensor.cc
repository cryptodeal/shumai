#include "napi_tensor.h"
#include <atomic>
#include <iostream>
#include <string>

using namespace Napi;

static std::atomic<size_t> g_bytes_used = 0;
static std::atomic<bool> g_row_major = true;

// non-exported helper functions
template <typename T>
std::vector<T> arrayArg(const void* ptr, int len, bool reverse, int invert) {
  std::vector<T> out;
  out.reserve(len);
  for (auto i = 0; i < len; ++i) {
    const auto idx = reverse ? len - i - 1 : i;
    auto v = reinterpret_cast<const int64_t*>(ptr)[idx];
    if (invert && v < 0) {
      v = -v - 1;
    } else if (invert) {
      v = invert - v - 1;
    }
    out.emplace_back(v);
  }
  return out;
}
template <typename T>
std::vector<T> jsArrayArg(Napi::Array arr,
                          bool reverse,
                          int invert,
                          Napi::Env env) {
  std::vector<T> out;
  const size_t len = static_cast<size_t>(arr.Length());
  out.reserve(len);
  for (size_t i = 0; i < len; ++i) {
    const auto idx = reverse ? len - i - 1 : i;
    Napi::Value val = arr[idx];
    if (!val.IsNumber()) {
      Napi::TypeError::New(env, "jsArrayArg requires `number[]`")
          .ThrowAsJavaScriptException();
      return out;
    } else {
      int64_t v = val.As<Napi::Number>().Int64Value();
      if (invert && v < 0) {
        v = -v - 1;
      } else if (invert) {
        v = invert - v - 1;
      }
      out.emplace_back(v);
    }
  }
  return out;
}

template <typename T>
std::vector<T> jsTensorArrayArg(Napi::Array arr, Napi::Env env) {
  std::vector<T> out;
  const size_t len = static_cast<size_t>(arr.Length());
  out.reserve(len);
  for (size_t i = 0; i < len; ++i) {
    Napi::Value temp = arr[i];
    if (temp.IsObject()) {
      Napi::Object tensor_obj = temp.As<Napi::Object>();
      if (tensor_obj.InstanceOf(Tensor::constructor->Value())) {
        Tensor* tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
        out.emplace_back(*(tensor->_tensor));
      } else {
        Napi::TypeError::New(env, "jsTensorArrayArg requires `Tensor[]`")
            .ThrowAsJavaScriptException();
        return out;
      }
    } else {
      Napi::TypeError::New(env, "jsTensorArrayArg requires `Tensor[]`")
          .ThrowAsJavaScriptException();
      return out;
    }
  }
  return out;
}

uint32_t axisArg(int32_t axis, bool reverse, int ndim) {
  if (!reverse) {
    return static_cast<uint32_t>(axis);
  }
  if (axis >= 0) {
    return static_cast<uint32_t>(ndim - axis - 1);
  } else {
    return static_cast<uint32_t>(-axis - 1);
  }
}

template <typename T>
std::vector<T> ptrArrayArg(const void* ptr, int len) {
  std::vector<T> out;
  out.reserve(len);
  for (auto i = 0; i < len; ++i) {
    auto ptrAsInt = reinterpret_cast<const int64_t*>(ptr)[i];
    auto ptr = reinterpret_cast<T*>(ptrAsInt);
    out.emplace_back(*ptr);
  }
  return out;
}

fl::Tensor* load(std::string filename, Napi::Env env) {
  try {
    fl::Tensor tensor;
    fl::load(filename, tensor);
    auto* t = new fl::Tensor(tensor);
    g_bytes_used += t->bytes();
    return t;
  } catch (std::exception const& e) {
    Napi::TypeError::New(env, e.what()).ThrowAsJavaScriptException();
  }
}

// global exported functions
static void init(const Napi::CallbackInfo& info) {
  fl::init();
}

static Napi::Value bytesUsed(const Napi::CallbackInfo& info) {
  return Napi::Number::New(info.Env(), g_bytes_used);
}

static void setRowMajor(const Napi::CallbackInfo& info) {
  g_row_major = true;
}

static void setColMajor(const Napi::CallbackInfo& info) {
  g_row_major = false;
}

static Napi::Value isRowMajor(const Napi::CallbackInfo& info) {
  return Napi::Boolean::New(info.Env(), g_row_major);
}

static Napi::Value isColMajor(const Napi::CallbackInfo& info) {
  return Napi::Boolean::New(info.Env(), !g_row_major);
}

static Napi::Value DtypeFloat32(const Napi::CallbackInfo& info) {
  return Napi::Number::New(info.Env(), static_cast<double>(fl::dtype::f32));
}

static Napi::Value DtypeFloat64(const Napi::CallbackInfo& info) {
  return Napi::Number::New(info.Env(), static_cast<double>(fl::dtype::f64));
}

static Napi::Value DtypeBoolInt8(const Napi::CallbackInfo& info) {
  return Napi::Number::New(info.Env(), static_cast<double>(fl::dtype::b8));
}

static Napi::Value DtypeInt16(const Napi::CallbackInfo& info) {
  return Napi::Number::New(info.Env(), static_cast<double>(fl::dtype::s16));
}

static Napi::Value DtypeInt32(const Napi::CallbackInfo& info) {
  return Napi::Number::New(info.Env(), static_cast<double>(fl::dtype::s32));
}

static Napi::Value DtypeInt64(const Napi::CallbackInfo& info) {
  return Napi::Number::New(info.Env(), static_cast<double>(fl::dtype::s64));
}

static Napi::Value DtypeUint8(const Napi::CallbackInfo& info) {
  return Napi::Number::New(info.Env(), static_cast<double>(fl::dtype::u8));
}

static Napi::Value DtypeUint16(const Napi::CallbackInfo& info) {
  return Napi::Number::New(info.Env(), static_cast<double>(fl::dtype::u16));
}

static Napi::Value DtypeUint32(const Napi::CallbackInfo& info) {
  return Napi::Number::New(info.Env(), static_cast<double>(fl::dtype::u32));
}

static Napi::Value DtypeUint64(const Napi::CallbackInfo& info) {
  return Napi::Number::New(info.Env(), static_cast<double>(fl::dtype::u64));
}

// global tensor operations
static Napi::Value Rand(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1 || !info[0].IsArray()) {
    Napi::TypeError::New(info.Env(),
                         "`rand` expects exactly 1 arg; "
                         "(expected type `number[]`)...")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  std::vector<long long> shape =
      jsArrayArg<long long>(info[0].As<Napi::Array>(), g_row_major, false, env);
  fl::Tensor t;
  t = fl::rand(fl::Shape(shape));
  g_bytes_used += t.bytes();
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

static Napi::Value Randn(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1 || !info[0].IsArray()) {
    Napi::TypeError::New(info.Env(),
                         "`randn` expects exactly 1 arg; "
                         "(expected type `number[]`)...")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  std::vector<long long> shape =
      jsArrayArg<long long>(info[0].As<Napi::Array>(), g_row_major, false, env);
  fl::Tensor t;
  t = fl::randn(fl::Shape(shape));
  g_bytes_used += t.bytes();
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

static Napi::Value Full(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2) {
    Napi::TypeError::New(info.Env(),
                         "`full` expects exactly 2 args; "
                         "(expected type `number[]`)...")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsArray()) {
    Napi::TypeError::New(
        info.Env(), "`full` expects 1st arg `shape` to be typeof `number[]`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[1].IsNumber()) {
    Napi::TypeError::New(info.Env(),
                         "`full` expects 2nd arg `val` to be typeof `number`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  std::vector<long long> shape =
      jsArrayArg<long long>(info[0].As<Napi::Array>(), g_row_major, false, env);
  float val = info[1].As<Napi::Number>().FloatValue();
  fl::Tensor t;
  t = fl::full(fl::Shape(shape), val);
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

static Napi::Value Identity(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1 || !info[0].IsNumber()) {
    Napi::TypeError::New(info.Env(),
                         "`identity` expects exactly 1 arg; "
                         "(expected type `number`)...")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  int64_t dim = info[1].As<Napi::Number>().Int64Value();
  fl::Tensor t;
  t = fl::identity(dim);
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

static Napi::Value Arange(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 3) {
    Napi::TypeError::New(info.Env(), "`arange` expects exactly 3 args...")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsNumber()) {
    Napi::TypeError::New(
        info.Env(), "`arange` expects 1st arg `start` to be typeof `number`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[1].IsNumber()) {
    Napi::TypeError::New(info.Env(),
                         "`arange` expects 2nd arg `end` to be typeof `number`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[2].IsNumber()) {
    Napi::TypeError::New(
        info.Env(), "`arange` expects 3rd arg `step` to be typeof `number`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  float start, end, step;
  start = info[0].As<Napi::Number>().FloatValue();
  end = info[1].As<Napi::Number>().FloatValue();
  step = info[2].As<Napi::Number>().FloatValue();
  fl::Tensor t;
  t = fl::arange(start, end, step);
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

static Napi::Value Iota(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2 || (!info[0].IsArray() || !info[1].IsArray())) {
    Napi::TypeError::New(
        info.Env(), "`iota` expects exactly 2 args; both of type `number[]`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto dims =
      jsArrayArg<long long>(info[0].As<Napi::Array>(), g_row_major, false, env);
  auto tileDims =
      jsArrayArg<long long>(info[1].As<Napi::Array>(), g_row_major, false, env);
  fl::Tensor t;
  t = fl::iota(fl::Shape(dims), fl::Shape(tileDims));
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

static Napi::Value Reshape(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2) {
    Napi::TypeError::New(info.Env(), "`reshape` expects exactly 2 args")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsObject()) {
    Napi::TypeError::New(
        info.Env(), "`reshape` expects 1st argument to be instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[1].IsArray()) {
    Napi::TypeError::New(
        info.Env(), "`reshape` expects 2nd argument to be typeof `number[]`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  auto shape =
      jsArrayArg<long long>(info[1].As<Napi::Array>(), g_row_major, false, env);
  if (tensor_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    fl::Tensor t;
    t = fl::reshape(*(parsed_tensor->_tensor), fl::Shape(shape));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value Transpose(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2) {
    Napi::TypeError::New(info.Env(), "`transpose` expects exactly 2 args")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsObject()) {
    Napi::TypeError::New(
        info.Env(),
        "`transpose` expects 1st argument to be instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[1].IsArray()) {
    Napi::TypeError::New(
        info.Env(), "`transpose` expects 2nd argument to be typeof `number[]`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    auto axes = jsArrayArg<long long>(info[1].As<Napi::Array>(), g_row_major,
                                      parsed_tensor->_tensor->ndim(), env);
    fl::Tensor t;
    t = fl::transpose(*(parsed_tensor->_tensor), fl::Shape(axes));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value Tile(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2) {
    Napi::TypeError::New(info.Env(), "`tile` expects exactly 2 args")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsObject()) {
    Napi::TypeError::New(
        info.Env(), "`tile` expects 1st argument to be instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[1].IsArray()) {
    Napi::TypeError::New(info.Env(),
                         "`tile` expects 2nd argument to be typeof `number[]`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  auto shape =
      jsArrayArg<long long>(info[1].As<Napi::Array>(), g_row_major, false, env);
  if (tensor_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    fl::Tensor t;
    t = fl::tile(*(parsed_tensor->_tensor), fl::Shape(shape));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value Concatenate(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2) {
    Napi::TypeError::New(info.Env(), "`concatenate` expects exactly 2 args")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsArray()) {
    Napi::TypeError::New(
        info.Env(),
        "`concatenate` expects 1st argument to be typeof `Tensor[]`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  const int num_tensors = info[0].As<Napi::Array>().Length();
  for (auto i = 0; i < num_tensors; ++i) {
    if (!info[0].As<Napi::Array>().Get(i).IsObject()) {
      Napi::TypeError::New(
          info.Env(),
          "`concatenate` expects 1st argument to be typeof `Tensor[]`")
          .ThrowAsJavaScriptException();
      return env.Null();
    }
  }
  if (!info[1].IsNumber()) {
    Napi::TypeError::New(
        info.Env(), "`concatenate` expects 2nd argument to be typeof `number`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto tensors = jsTensorArrayArg<fl::Tensor>(info[0].As<Napi::Array>(), env);
  auto used_axis = axisArg(info[1].As<Napi::Number>().Int32Value(), g_row_major,
                           (&tensors[0])->ndim());
  fl::Tensor t;
  t = fl::concatenate(tensors, used_axis);
  g_bytes_used += t.bytes();
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

static Napi::Value NonZero(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1) {
    Napi::TypeError::New(info.Env(), "`nonzero` expects exactly 1 arg")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsObject()) {
    Napi::TypeError::New(
        info.Env(), "`nonzero` expects 1st argument to be instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    fl::Tensor t;
    t = fl::nonzero(*(parsed_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value Negative(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1) {
    Napi::TypeError::New(info.Env(), "`negative` expects exactly 2 args")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsObject()) {
    Napi::TypeError::New(
        info.Env(), "`negative` expects 1st argument to be instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    fl::Tensor t;
    t = fl::negative(*(parsed_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value LogicalNot(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1) {
    Napi::TypeError::New(info.Env(), "`logicalNot` expects exactly 2 args")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsObject()) {
    Napi::TypeError::New(
        info.Env(),
        "`logicalNot` expects 1st argument to be instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    fl::Tensor t;
    t = fl::logicalNot(*(parsed_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value Exp(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1) {
    Napi::TypeError::New(info.Env(), "`exp` expects exactly 2 args")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsObject()) {
    Napi::TypeError::New(info.Env(),
                         "`exp` expects 1st argument to be instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    fl::Tensor t;
    t = fl::exp(*(parsed_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value Log(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1) {
    Napi::TypeError::New(info.Env(), "`log` expects exactly 2 args")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsObject()) {
    Napi::TypeError::New(info.Env(),
                         "`log` expects 1st argument to be instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    fl::Tensor t;
    t = fl::log(*(parsed_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value Log1p(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1) {
    Napi::TypeError::New(info.Env(), "`log1p` expects exactly 2 args")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsObject()) {
    Napi::TypeError::New(
        info.Env(), "`log1p` expects 1st argument to be instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    fl::Tensor t;
    t = fl::log1p(*(parsed_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value Sin(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1) {
    Napi::TypeError::New(info.Env(), "`sin` expects exactly 2 args")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsObject()) {
    Napi::TypeError::New(info.Env(),
                         "`sin` expects 1st argument to be instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    fl::Tensor t;
    t = fl::sin(*(parsed_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value Cos(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1) {
    Napi::TypeError::New(info.Env(), "`cos` expects exactly 1 arg")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsObject()) {
    Napi::TypeError::New(info.Env(),
                         "`cos` expects 1st argument to be instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    fl::Tensor t;
    t = fl::cos(*(parsed_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value Sqrt(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1) {
    Napi::TypeError::New(info.Env(), "`sqrt` expects exactly 2 args")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsObject()) {
    Napi::TypeError::New(
        info.Env(), "`sqrt` expects 1st argument to be instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    fl::Tensor t;
    t = fl::sqrt(*(parsed_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value Tanh(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1) {
    Napi::TypeError::New(info.Env(), "`tanh` expects exactly 2 args")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsObject()) {
    Napi::TypeError::New(
        info.Env(), "`tanh` expects 1st argument to be instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    fl::Tensor t;
    t = fl::tanh(*(parsed_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value Floor(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1) {
    Napi::TypeError::New(info.Env(), "`floor` expects exactly 2 args")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsObject()) {
    Napi::TypeError::New(
        info.Env(), "`floor` expects 1st argument to be instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    fl::Tensor t;
    t = fl::floor(*(parsed_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value Ceil(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1) {
    Napi::TypeError::New(info.Env(), "`ceil` expects exactly 2 args")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsObject()) {
    Napi::TypeError::New(
        info.Env(), "`ceil` expects 1st argument to be instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    fl::Tensor t;
    t = fl::ceil(*(parsed_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value Rint(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1) {
    Napi::TypeError::New(info.Env(), "`rint` expects exactly 2 args")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsObject()) {
    Napi::TypeError::New(
        info.Env(), "`rint` expects 1st argument to be instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    fl::Tensor t;
    t = fl::rint(*(parsed_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value Absolute(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1) {
    Napi::TypeError::New(info.Env(), "`rint` expects exactly 2 args")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsObject()) {
    Napi::TypeError::New(
        info.Env(), "`rint` expects 1st argument to be instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    fl::Tensor t;
    t = fl::absolute(*(parsed_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value Sigmoid(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1) {
    Napi::TypeError::New(info.Env(), "`rint` expects exactly 2 args")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsObject()) {
    Napi::TypeError::New(
        info.Env(), "`rint` expects 1st argument to be instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    fl::Tensor t;
    t = fl::sigmoid(*(parsed_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value Erf(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1) {
    Napi::TypeError::New(info.Env(), "`rint` expects exactly 2 args")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsObject()) {
    Napi::TypeError::New(
        info.Env(), "`rint` expects 1st argument to be instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    fl::Tensor t;
    t = fl::erf(*(parsed_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value Flip(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2) {
    Napi::TypeError::New(info.Env(), "`flip` expects exactly 2 args")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsObject()) {
    Napi::TypeError::New(
        info.Env(), "`flip` expects 1st argument to be instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[1].IsNumber()) {
    Napi::TypeError::New(info.Env(),
                         "`flip` expects 2nd argument to be typeof `number`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  uint32_t dim = info[1].As<Napi::Number>().Uint32Value();
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    fl::Tensor t;
    t = fl::flip(*(parsed_tensor->_tensor), dim);
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value Clip(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 3 ||
      (!info[0].IsObject() && !info[1].IsObject() && !info[2].IsObject())) {
    Napi::TypeError::New(
        info.Env(), "`clip` expects exactly 3 args, each instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  Napi::Object low_obj = info[1].As<Napi::Object>();
  Napi::Object high_obj = info[2].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value()) &&
      low_obj.InstanceOf(Tensor::constructor->Value()) &&
      high_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    Tensor* parsed_low = Napi::ObjectWrap<Tensor>::Unwrap(low_obj);
    Tensor* parsed_high = Napi::ObjectWrap<Tensor>::Unwrap(high_obj);
    fl::Tensor t;
    t = fl::clip(*(parsed_tensor->_tensor), *(parsed_low->_tensor),
                 *(parsed_high->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value Roll(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 3) {
    Napi::TypeError::New(info.Env(), "`roll` expects exactly 3 args")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsObject()) {
    Napi::TypeError::New(
        info.Env(), "`roll` expects 1st argument to be instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[1].IsNumber()) {
    Napi::TypeError::New(info.Env(),
                         "`roll` expects 2nd argument to be typeof `number`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[2].IsNumber()) {
    Napi::TypeError::New(info.Env(),
                         "`roll` expects 3rd argument to be typeof `number`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  int shift = static_cast<int>(info[1].As<Napi::Number>().Int64Value());
  if (tensor_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    auto used_axis = axisArg(info[2].As<Napi::Number>().Uint32Value(),
                             g_row_major, parsed_tensor->_tensor->ndim());
    fl::Tensor t;
    t = fl::roll(*(parsed_tensor->_tensor), shift, used_axis);
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value IsNaN(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1) {
    Napi::TypeError::New(info.Env(), "`isnan` expects exactly 2 args")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsObject()) {
    Napi::TypeError::New(
        info.Env(), "`isnan` expects 1st argument to be instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    fl::Tensor t;
    t = fl::isnan(*(parsed_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value IsInf(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1) {
    Napi::TypeError::New(info.Env(), "`isinf` expects exactly 2 args")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsObject()) {
    Napi::TypeError::New(
        info.Env(), "`isinf` expects 1st argument to be instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    fl::Tensor t;
    t = fl::isinf(*(parsed_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value Sign(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1) {
    Napi::TypeError::New(info.Env(), "`isinf` expects exactly 2 args")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsObject()) {
    Napi::TypeError::New(
        info.Env(), "`isinf` expects 1st argument to be instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    fl::Tensor t;
    t = fl::sign(*(parsed_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value Tril(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1) {
    Napi::TypeError::New(info.Env(), "`isinf` expects exactly 2 args")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsObject()) {
    Napi::TypeError::New(
        info.Env(), "`isinf` expects 1st argument to be instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    fl::Tensor t;
    if (g_row_major) {
      t = fl::triu(*(parsed_tensor->_tensor));
    } else {
      t = fl::tril(*(parsed_tensor->_tensor));
    }
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value Triu(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1) {
    Napi::TypeError::New(info.Env(), "`isinf` expects exactly 2 args")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsObject()) {
    Napi::TypeError::New(
        info.Env(), "`isinf` expects 1st argument to be instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    fl::Tensor t;
    if (g_row_major) {
      t = fl::tril(*(parsed_tensor->_tensor));
    } else {
      t = fl::triu(*(parsed_tensor->_tensor));
    }
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value Where(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 3 ||
      (!info[0].IsObject() && !info[1].IsObject() && !info[2].IsObject())) {
    Napi::TypeError::New(
        info.Env(), "`where` expects exactly 3 args, each instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object cond_obj = info[0].As<Napi::Object>();
  Napi::Object x_obj = info[1].As<Napi::Object>();
  Napi::Object y_obj = info[2].As<Napi::Object>();
  if (cond_obj.InstanceOf(Tensor::constructor->Value()) &&
      x_obj.InstanceOf(Tensor::constructor->Value()) &&
      y_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* cond_tensor = Napi::ObjectWrap<Tensor>::Unwrap(cond_obj);
    Tensor* x_tensor = Napi::ObjectWrap<Tensor>::Unwrap(x_obj);
    Tensor* y_tensor = Napi::ObjectWrap<Tensor>::Unwrap(y_obj);
    fl::Tensor t;
    t = fl::where(cond_tensor->_tensor->astype(fl::dtype::b8),
                  *(x_tensor->_tensor), *(y_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value Sort(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2) {
    Napi::TypeError::New(info.Env(), "`sort` expects exactly 2 args")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsObject()) {
    Napi::TypeError::New(
        info.Env(), "`sort` expects 1st argument to be instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[1].IsNumber()) {
    Napi::TypeError::New(info.Env(),
                         "`sort` expects 2nd argument to be typeof `number`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  uint32_t dim = info[1].As<Napi::Number>().Uint32Value();
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    fl::Tensor t;
    t = fl::sort(*(parsed_tensor->_tensor), dim);
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value Add(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2 || (!info[0].IsObject() && !info[1].IsObject())) {
    Napi::TypeError::New(
        info.Env(), "`add` expects exactly 2 args, each instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  Napi::Object other_obj = info[1].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value()) &&
      other_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    Tensor* other_tensor = Napi::ObjectWrap<Tensor>::Unwrap(other_obj);
    fl::Tensor t;
    t = fl::add(*(parsed_tensor->_tensor), *(other_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value Sub(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2 || (!info[0].IsObject() && !info[1].IsObject())) {
    Napi::TypeError::New(
        info.Env(), "`sub` expects exactly 2 args, each instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  Napi::Object other_obj = info[1].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value()) &&
      other_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    Tensor* other_tensor = Napi::ObjectWrap<Tensor>::Unwrap(other_obj);
    fl::Tensor t;
    t = fl::sub(*(parsed_tensor->_tensor), *(other_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value Mul(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2 || (!info[0].IsObject() && !info[1].IsObject())) {
    Napi::TypeError::New(
        info.Env(), "`mul` expects exactly 2 args, each instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  Napi::Object other_obj = info[1].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value()) &&
      other_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    Tensor* other_tensor = Napi::ObjectWrap<Tensor>::Unwrap(other_obj);
    fl::Tensor t;
    t = fl::mul(*(parsed_tensor->_tensor), *(other_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value Div(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2 || (!info[0].IsObject() && !info[1].IsObject())) {
    Napi::TypeError::New(
        info.Env(), "`div` expects exactly 2 args, each instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  Napi::Object other_obj = info[1].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value()) &&
      other_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    Tensor* other_tensor = Napi::ObjectWrap<Tensor>::Unwrap(other_obj);
    fl::Tensor t;
    t = fl::div(*(parsed_tensor->_tensor), *(other_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value Eq(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2 || (!info[0].IsObject() && !info[1].IsObject())) {
    Napi::TypeError::New(
        info.Env(), "`eq` expects exactly 2 args, each instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  Napi::Object other_obj = info[1].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value()) &&
      other_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    Tensor* other_tensor = Napi::ObjectWrap<Tensor>::Unwrap(other_obj);
    fl::Tensor t;
    t = fl::eq(*(parsed_tensor->_tensor), *(other_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value Neq(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2 || (!info[0].IsObject() && !info[1].IsObject())) {
    Napi::TypeError::New(
        info.Env(), "`neq` expects exactly 2 args, each instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  Napi::Object other_obj = info[1].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value()) &&
      other_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    Tensor* other_tensor = Napi::ObjectWrap<Tensor>::Unwrap(other_obj);
    fl::Tensor t;
    t = fl::neq(*(parsed_tensor->_tensor), *(other_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value LessThan(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2 || (!info[0].IsObject() && !info[1].IsObject())) {
    Napi::TypeError::New(
        info.Env(),
        "`lessThan` expects exactly 2 args, each instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  Napi::Object other_obj = info[1].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value()) &&
      other_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    Tensor* other_tensor = Napi::ObjectWrap<Tensor>::Unwrap(other_obj);
    fl::Tensor t;
    t = fl::lessThan(*(parsed_tensor->_tensor), *(other_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value LessThanEqual(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2 || (!info[0].IsObject() && !info[1].IsObject())) {
    Napi::TypeError::New(
        info.Env(),
        "`lessThanEqual` expects exactly 2 args, each instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  Napi::Object other_obj = info[1].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value()) &&
      other_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    Tensor* other_tensor = Napi::ObjectWrap<Tensor>::Unwrap(other_obj);
    fl::Tensor t;
    t = fl::lessThanEqual(*(parsed_tensor->_tensor), *(other_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value GreaterThan(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2 || (!info[0].IsObject() && !info[1].IsObject())) {
    Napi::TypeError::New(
        info.Env(),
        "`greaterThan` expects exactly 2 args, each instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  Napi::Object other_obj = info[1].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value()) &&
      other_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    Tensor* other_tensor = Napi::ObjectWrap<Tensor>::Unwrap(other_obj);
    fl::Tensor t;
    t = fl::greaterThan(*(parsed_tensor->_tensor), *(other_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value GreaterThanEqual(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2 || (!info[0].IsObject() && !info[1].IsObject())) {
    Napi::TypeError::New(
        info.Env(),
        "`greaterThanEqual` expects exactly 2 args, each instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  Napi::Object other_obj = info[1].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value()) &&
      other_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    Tensor* other_tensor = Napi::ObjectWrap<Tensor>::Unwrap(other_obj);
    fl::Tensor t;
    t = fl::greaterThanEqual(*(parsed_tensor->_tensor),
                             *(other_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value LogicalOr(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2 || (!info[0].IsObject() && !info[1].IsObject())) {
    Napi::TypeError::New(
        info.Env(),
        "`logicalOr` expects exactly 2 args, each instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  Napi::Object other_obj = info[1].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value()) &&
      other_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    Tensor* other_tensor = Napi::ObjectWrap<Tensor>::Unwrap(other_obj);
    fl::Tensor t;
    t = fl::logicalOr(*(parsed_tensor->_tensor), *(other_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value LogicalAnd(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2 || (!info[0].IsObject() && !info[1].IsObject())) {
    Napi::TypeError::New(
        info.Env(),
        "`logicalAnd` expects exactly 2 args, each instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  Napi::Object other_obj = info[1].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value()) &&
      other_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    Tensor* other_tensor = Napi::ObjectWrap<Tensor>::Unwrap(other_obj);
    fl::Tensor t;
    t = fl::logicalAnd(*(parsed_tensor->_tensor), *(other_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value Mod(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2 || (!info[0].IsObject() && !info[1].IsObject())) {
    Napi::TypeError::New(
        info.Env(), "`mod` expects exactly 2 args, each instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  Napi::Object other_obj = info[1].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value()) &&
      other_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    Tensor* other_tensor = Napi::ObjectWrap<Tensor>::Unwrap(other_obj);
    fl::Tensor t;
    t = fl::mod(*(parsed_tensor->_tensor), *(other_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value BitwiseAnd(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2 || (!info[0].IsObject() && !info[1].IsObject())) {
    Napi::TypeError::New(
        info.Env(),
        "`bitwiseAnd` expects exactly 2 args, each instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  Napi::Object other_obj = info[1].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value()) &&
      other_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    Tensor* other_tensor = Napi::ObjectWrap<Tensor>::Unwrap(other_obj);
    fl::Tensor t;
    t = fl::bitwiseAnd(*(parsed_tensor->_tensor), *(other_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value BitwiseOr(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2 || (!info[0].IsObject() && !info[1].IsObject())) {
    Napi::TypeError::New(
        info.Env(),
        "`bitwiseOr` expects exactly 2 args, each instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  Napi::Object other_obj = info[1].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value()) &&
      other_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    Tensor* other_tensor = Napi::ObjectWrap<Tensor>::Unwrap(other_obj);
    fl::Tensor t;
    t = fl::bitwiseOr(*(parsed_tensor->_tensor), *(other_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value BitwiseXor(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2 || (!info[0].IsObject() && !info[1].IsObject())) {
    Napi::TypeError::New(
        info.Env(),
        "`bitwiseXor` expects exactly 2 args, each instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  Napi::Object other_obj = info[1].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value()) &&
      other_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    Tensor* other_tensor = Napi::ObjectWrap<Tensor>::Unwrap(other_obj);
    fl::Tensor t;
    t = fl::bitwiseXor(*(parsed_tensor->_tensor), *(other_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value LShift(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2 || (!info[0].IsObject() && !info[1].IsObject())) {
    Napi::TypeError::New(
        info.Env(), "`lShift` expects exactly 2 args, each instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  Napi::Object other_obj = info[1].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value()) &&
      other_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    Tensor* other_tensor = Napi::ObjectWrap<Tensor>::Unwrap(other_obj);
    fl::Tensor t;
    t = fl::lShift(*(parsed_tensor->_tensor), *(other_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value RShift(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2 || (!info[0].IsObject() && !info[1].IsObject())) {
    Napi::TypeError::New(
        info.Env(), "`rShift` expects exactly 2 args, each instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  Napi::Object other_obj = info[1].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value()) &&
      other_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    Tensor* other_tensor = Napi::ObjectWrap<Tensor>::Unwrap(other_obj);
    fl::Tensor t;
    t = fl::rShift(*(parsed_tensor->_tensor), *(other_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value Minimum(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2 || (!info[0].IsObject() && !info[1].IsObject())) {
    Napi::TypeError::New(
        info.Env(),
        "`minimum` expects exactly 2 args, each instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  Napi::Object other_obj = info[1].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value()) &&
      other_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    Tensor* other_tensor = Napi::ObjectWrap<Tensor>::Unwrap(other_obj);
    fl::Tensor t;
    t = fl::minimum(*(parsed_tensor->_tensor), *(other_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value Maximum(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2 || (!info[0].IsObject() && !info[1].IsObject())) {
    Napi::TypeError::New(
        info.Env(),
        "`maximum` expects exactly 2 args, each instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  Napi::Object other_obj = info[1].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value()) &&
      other_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    Tensor* other_tensor = Napi::ObjectWrap<Tensor>::Unwrap(other_obj);
    fl::Tensor t;
    t = fl::maximum(*(parsed_tensor->_tensor), *(other_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value Power(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2 || (!info[0].IsObject() && !info[1].IsObject())) {
    Napi::TypeError::New(
        info.Env(), "`power` expects exactly 2 args, each instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  Napi::Object other_obj = info[1].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value()) &&
      other_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    Tensor* other_tensor = Napi::ObjectWrap<Tensor>::Unwrap(other_obj);
    fl::Tensor t;
    t = fl::power(*(parsed_tensor->_tensor), *(other_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value MatMul(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2 || (!info[0].IsObject() && !info[1].IsObject())) {
    Napi::TypeError::New(
        info.Env(), "`matmul` expects exactly 2 args, each instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  Napi::Object other_obj = info[1].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value()) &&
      other_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    Tensor* other_tensor = Napi::ObjectWrap<Tensor>::Unwrap(other_obj);
    fl::Tensor t;
    if (g_row_major) {
      t = fl::matmul(*(other_tensor->_tensor), *(parsed_tensor->_tensor));
    } else {
      t = fl::matmul(*(parsed_tensor->_tensor), *(other_tensor->_tensor));
    }
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value Conv2d(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  if (info.Length() >= 2 || (!info[0].IsObject() || !info[1].IsObject())) {
    Napi::TypeError::New(info.Env(),
                         "Tensor method `conv2d` expects minimum 2 args "
                         "both of type Tensor")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  int32_t sx, sy, px, py, dx, dy, groups;
  if (info.Length() >= 3 && info[2].IsNumber()) {
    sx = info[2].As<Napi::Number>().Int32Value();
  } else {
    sx = 1;
  }
  if (info.Length() >= 4 && info[3].IsNumber()) {
    sy = info[3].As<Napi::Number>().Int32Value();
  } else {
    sy = 1;
  }
  if (info.Length() >= 5 && info[4].IsNumber()) {
    px = info[4].As<Napi::Number>().Int32Value();
  } else {
    px = 0;
  }
  if (info.Length() >= 6 && info[5].IsNumber()) {
    py = info[5].As<Napi::Number>().Int32Value();
  } else {
    py = 0;
  }
  if (info.Length() >= 7 && info[6].IsNumber()) {
    dx = info[6].As<Napi::Number>().Int32Value();
  } else {
    dx = 1;
  }
  if (info.Length() >= 8 && info[7].IsNumber()) {
    dy = info[7].As<Napi::Number>().Int32Value();
  } else {
    dy = 1;
  }
  if (info.Length() >= 9 && info[8].IsNumber()) {
    groups = info[8].As<Napi::Number>().Int32Value();
  } else {
    groups = 1;
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  Napi::Object weights_obj = info[1].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value()) &&
      weights_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    Tensor* weights_tensor = Napi::ObjectWrap<Tensor>::Unwrap(weights_obj);
    fl::Tensor t;
    t = fl::conv2d(*(parsed_tensor->_tensor), *(weights_tensor->_tensor), sx,
                   sy, px, py, dx, dy, groups);
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value AMin(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 3) {
    Napi::TypeError::New(info.Env(), "`amin` expects exactly 3 args")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsObject()) {
    Napi::TypeError::New(
        info.Env(), "`amin` expects 1st argument to be instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[1].IsArray()) {
    Napi::TypeError::New(info.Env(),
                         "`amin` expects 2nd argument to be typeof `number[]`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[2].IsBoolean()) {
    Napi::TypeError::New(info.Env(),
                         "`amin` expects 3rd argument to be typeof `boolean`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  bool keep_dims = info[2].As<Napi::Boolean>().Value();
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    auto axes = jsArrayArg<int>(info[1].As<Napi::Array>(), g_row_major,
                                parsed_tensor->_tensor->ndim(), env);
    fl::Tensor t;
    t = fl::amin(*(parsed_tensor->_tensor), axes, keep_dims);
    auto axes_set = std::unordered_set<int>(axes.begin(), axes.end());
    auto base_shape = parsed_tensor->_tensor->shape().get();
    std::vector<fl::Dim> new_shape;
    for (size_t idx = 0; idx < base_shape.size(); ++idx) {
      if (axes_set.count(idx) || (axes_set.size() == 0)) {
        if (keep_dims) {
          new_shape.emplace_back(1);
        }
        continue;
      }
      new_shape.emplace_back(base_shape[idx]);
    }
    const auto& shape = fl::Shape(new_shape);
    t = fl::reshape(t, shape);
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value AMax(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 3) {
    Napi::TypeError::New(info.Env(), "`amax` expects exactly 3 args")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsObject()) {
    Napi::TypeError::New(
        info.Env(), "`amax` expects 1st argument to be instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[1].IsArray()) {
    Napi::TypeError::New(info.Env(),
                         "`amax` expects 2nd argument to be typeof `number[]`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[2].IsBoolean()) {
    Napi::TypeError::New(info.Env(),
                         "`amax` expects 3rd argument to be typeof `boolean`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  bool keep_dims = info[2].As<Napi::Boolean>().Value();
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    auto axes = jsArrayArg<int>(info[1].As<Napi::Array>(), g_row_major,
                                parsed_tensor->_tensor->ndim(), env);
    fl::Tensor t;
    t = fl::amax(*(parsed_tensor->_tensor), axes, keep_dims);
    auto axes_set = std::unordered_set<int>(axes.begin(), axes.end());
    auto base_shape = parsed_tensor->_tensor->shape().get();
    std::vector<fl::Dim> new_shape;
    for (size_t idx = 0; idx < base_shape.size(); ++idx) {
      if (axes_set.count(idx) || (axes_set.size() == 0)) {
        if (keep_dims) {
          new_shape.emplace_back(1);
        }
        continue;
      }
      new_shape.emplace_back(base_shape[idx]);
    }
    const auto& shape = fl::Shape(new_shape);
    t = fl::reshape(t, shape);
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value ArgMin(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 3) {
    Napi::TypeError::New(info.Env(), "`argmin` expects exactly 3 args")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsObject()) {
    Napi::TypeError::New(
        info.Env(), "`argmin` expects 1st argument to be instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[1].IsNumber()) {
    Napi::TypeError::New(info.Env(),
                         "`argmin` expects 2nd argument to be typeof `number`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[2].IsBoolean()) {
    Napi::TypeError::New(info.Env(),
                         "`argmin` expects 3rd argument to be typeof `boolean`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  uint32_t axis = info[1].As<Napi::Number>().Uint32Value();
  bool keep_dims = info[2].As<Napi::Boolean>().Value();
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    auto used_axis = axisArg(axis, g_row_major, parsed_tensor->_tensor->ndim());
    fl::Tensor t;
    t = fl::argmin(*(parsed_tensor->_tensor), used_axis, keep_dims);
    auto axes_set = std::unordered_set<int>{static_cast<int>(used_axis)};
    auto base_shape = parsed_tensor->_tensor->shape().get();
    std::vector<fl::Dim> new_shape;
    for (size_t idx = 0; idx < base_shape.size(); ++idx) {
      if (axes_set.count(idx) || (axes_set.size() == 0)) {
        if (keep_dims) {
          new_shape.emplace_back(1);
        }
        continue;
      }
      new_shape.emplace_back(base_shape[idx]);
    }
    const auto& shape = fl::Shape(new_shape);
    t = fl::reshape(t, shape);
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value ArgMax(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 3) {
    Napi::TypeError::New(info.Env(), "`argmax` expects exactly 3 args")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsObject()) {
    Napi::TypeError::New(
        info.Env(), "`argmax` expects 1st argument to be instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[1].IsNumber()) {
    Napi::TypeError::New(info.Env(),
                         "`argmax` expects 2nd argument to be typeof `number`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[2].IsBoolean()) {
    Napi::TypeError::New(info.Env(),
                         "`argmax` expects 3rd argument to be typeof `boolean`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  uint32_t axis = info[1].As<Napi::Number>().Uint32Value();
  bool keep_dims = info[2].As<Napi::Boolean>().Value();
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    auto used_axis = axisArg(axis, g_row_major, parsed_tensor->_tensor->ndim());
    fl::Tensor t;
    t = fl::argmax(*(parsed_tensor->_tensor), used_axis, keep_dims);
    auto axes_set = std::unordered_set<int>{static_cast<int>(used_axis)};
    auto base_shape = parsed_tensor->_tensor->shape().get();
    std::vector<fl::Dim> new_shape;
    for (size_t idx = 0; idx < base_shape.size(); ++idx) {
      if (axes_set.count(idx) || (axes_set.size() == 0)) {
        if (keep_dims) {
          new_shape.emplace_back(1);
        }
        continue;
      }
      new_shape.emplace_back(base_shape[idx]);
    }
    const auto& shape = fl::Shape(new_shape);
    t = fl::reshape(t, shape);
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value Sum(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 3) {
    Napi::TypeError::New(info.Env(), "`sum` expects exactly 3 args")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsObject()) {
    Napi::TypeError::New(info.Env(),
                         "`sum` expects 1st argument to be instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[1].IsArray()) {
    Napi::TypeError::New(info.Env(),
                         "`sum` expects 2nd argument to be typeof `number[]`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[2].IsBoolean()) {
    Napi::TypeError::New(info.Env(),
                         "`sum` expects 3rd argument to be typeof `boolean`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  bool keep_dims = info[2].As<Napi::Boolean>().Value();
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    auto axes = jsArrayArg<int>(info[1].As<Napi::Array>(), g_row_major,
                                parsed_tensor->_tensor->ndim(), env);
    fl::Tensor t;
    t = fl::sum(*(parsed_tensor->_tensor), axes, keep_dims);
    auto axes_set = std::unordered_set<int>(axes.begin(), axes.end());
    auto base_shape = parsed_tensor->_tensor->shape().get();
    std::vector<fl::Dim> new_shape;
    for (size_t idx = 0; idx < base_shape.size(); ++idx) {
      if (axes_set.count(idx) || (axes_set.size() == 0)) {
        if (keep_dims) {
          new_shape.emplace_back(1);
        }
        continue;
      }
      new_shape.emplace_back(base_shape[idx]);
    }
    const auto& shape = fl::Shape(new_shape);
    t = fl::reshape(t, shape);
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value CumSum(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2) {
    Napi::TypeError::New(info.Env(), "`cumsum` expects exactly 2 args")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsObject()) {
    Napi::TypeError::New(
        info.Env(), "`cumsum` expects 1st argument to be instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[1].IsNumber()) {
    Napi::TypeError::New(info.Env(),
                         "`cumsum` expects 2nd argument to be typeof `number`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  uint32_t axis = info[1].As<Napi::Number>().Uint32Value();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    auto used_axis = axisArg(axis, g_row_major, parsed_tensor->_tensor->ndim());
    fl::Tensor t;
    t = fl::cumsum(*(parsed_tensor->_tensor), used_axis);
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value Mean(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 3) {
    Napi::TypeError::New(info.Env(), "`mean` expects exactly 3 args")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsObject()) {
    Napi::TypeError::New(
        info.Env(), "`mean` expects 1st argument to be instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[1].IsArray()) {
    Napi::TypeError::New(info.Env(),
                         "`mean` expects 2nd argument to be typeof `number[]`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[2].IsBoolean()) {
    Napi::TypeError::New(info.Env(),
                         "`mean` expects 3rd argument to be typeof `boolean`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  bool keep_dims = info[2].As<Napi::Boolean>().Value();
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    auto axes = jsArrayArg<int>(info[1].As<Napi::Array>(), g_row_major,
                                parsed_tensor->_tensor->ndim(), env);
    fl::Tensor t;
    t = fl::mean(*(parsed_tensor->_tensor), axes, keep_dims);
    auto axes_set = std::unordered_set<int>(axes.begin(), axes.end());
    auto base_shape = parsed_tensor->_tensor->shape().get();
    std::vector<fl::Dim> new_shape;
    for (size_t idx = 0; idx < base_shape.size(); ++idx) {
      if (axes_set.count(idx) || (axes_set.size() == 0)) {
        if (keep_dims) {
          new_shape.emplace_back(1);
        }
        continue;
      }
      new_shape.emplace_back(base_shape[idx]);
    }
    const auto& shape = fl::Shape(new_shape);
    t = fl::reshape(t, shape);
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value Median(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 3) {
    Napi::TypeError::New(info.Env(), "`median` expects exactly 3 args")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsObject()) {
    Napi::TypeError::New(
        info.Env(), "`median` expects 1st argument to be instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[1].IsArray()) {
    Napi::TypeError::New(
        info.Env(), "`median` expects 2nd argument to be typeof `number[]`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[2].IsBoolean()) {
    Napi::TypeError::New(info.Env(),
                         "`median` expects 3rd argument to be typeof `boolean`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  bool keep_dims = info[2].As<Napi::Boolean>().Value();
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    auto axes = jsArrayArg<int>(info[1].As<Napi::Array>(), g_row_major,
                                parsed_tensor->_tensor->ndim(), env);
    fl::Tensor t;
    t = fl::median(*(parsed_tensor->_tensor), axes, keep_dims);
    auto axes_set = std::unordered_set<int>(axes.begin(), axes.end());
    auto base_shape = parsed_tensor->_tensor->shape().get();
    std::vector<fl::Dim> new_shape;
    for (size_t idx = 0; idx < base_shape.size(); ++idx) {
      if (axes_set.count(idx) || (axes_set.size() == 0)) {
        if (keep_dims) {
          new_shape.emplace_back(1);
        }
        continue;
      }
      new_shape.emplace_back(base_shape[idx]);
    }
    const auto& shape = fl::Shape(new_shape);
    t = fl::reshape(t, shape);
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value Var(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 4) {
    Napi::TypeError::New(info.Env(), "`var` expects exactly 3 args")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsObject()) {
    Napi::TypeError::New(info.Env(),
                         "`var` expects 1st argument to be instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[1].IsArray()) {
    Napi::TypeError::New(info.Env(),
                         "`var` expects 2nd argument to be typeof `number[]`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[2].IsBoolean()) {
    Napi::TypeError::New(info.Env(),
                         "`var` expects 3rd argument to be typeof `boolean`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[3].IsBoolean()) {
    Napi::TypeError::New(info.Env(),
                         "`var` expects 3rd argument to be typeof `boolean`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  bool bias = info[2].As<Napi::Boolean>().Value();
  bool keep_dims = info[3].As<Napi::Boolean>().Value();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    auto axes = jsArrayArg<int>(info[1].As<Napi::Array>(), g_row_major,
                                parsed_tensor->_tensor->ndim(), env);
    fl::Tensor t;
    t = fl::var(*(parsed_tensor->_tensor), axes, bias, keep_dims);
    auto axes_set = std::unordered_set<int>(axes.begin(), axes.end());
    auto base_shape = parsed_tensor->_tensor->shape().get();
    std::vector<fl::Dim> new_shape;
    for (size_t idx = 0; idx < base_shape.size(); ++idx) {
      if (axes_set.count(idx) || (axes_set.size() == 0)) {
        if (keep_dims) {
          new_shape.emplace_back(1);
        }
        continue;
      }
      new_shape.emplace_back(base_shape[idx]);
    }
    const auto& shape = fl::Shape(new_shape);
    t = fl::reshape(t, shape);
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value Std(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 3) {
    Napi::TypeError::New(info.Env(), "`std` expects exactly 3 args")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsObject()) {
    Napi::TypeError::New(info.Env(),
                         "`std` expects 1st argument to be instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[1].IsArray()) {
    Napi::TypeError::New(info.Env(),
                         "`std` expects 2nd argument to be typeof `number[]`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[2].IsBoolean()) {
    Napi::TypeError::New(info.Env(),
                         "`std` expects 3rd argument to be typeof `boolean`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  bool keep_dims = info[2].As<Napi::Boolean>().Value();
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    auto axes = jsArrayArg<int>(info[1].As<Napi::Array>(), g_row_major,
                                parsed_tensor->_tensor->ndim(), env);
    fl::Tensor t;
    t = fl::std(*(parsed_tensor->_tensor), axes, keep_dims);
    auto axes_set = std::unordered_set<int>(axes.begin(), axes.end());
    auto base_shape = parsed_tensor->_tensor->shape().get();
    std::vector<fl::Dim> new_shape;
    for (size_t idx = 0; idx < base_shape.size(); ++idx) {
      if (axes_set.count(idx) || (axes_set.size() == 0)) {
        if (keep_dims) {
          new_shape.emplace_back(1);
        }
        continue;
      }
      new_shape.emplace_back(base_shape[idx]);
    }
    const auto& shape = fl::Shape(new_shape);
    t = fl::reshape(t, shape);
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value Norm(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 4) {
    Napi::TypeError::New(info.Env(), "`norm` expects exactly 4 args")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsObject()) {
    Napi::TypeError::New(
        info.Env(), "`norm` expects 1st argument to be instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[1].IsArray()) {
    Napi::TypeError::New(info.Env(),
                         "`norm` expects 2nd argument to be typeof `number[]`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[2].IsNumber()) {
    Napi::TypeError::New(info.Env(),
                         "`norm` expects 3rd argument to be typeof `number`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[3].IsBoolean()) {
    Napi::TypeError::New(info.Env(),
                         "`norm` expects 4th argument to be typeof `boolean`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  double p = info[2].As<Napi::Number>().DoubleValue();
  bool keep_dims = info[3].As<Napi::Boolean>().Value();
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    auto axes = jsArrayArg<int>(info[1].As<Napi::Array>(), g_row_major,
                                parsed_tensor->_tensor->ndim(), env);
    fl::Tensor t;
    t = fl::norm(*(parsed_tensor->_tensor), axes, p, keep_dims);
    if (p == std::numeric_limits<double>::infinity()) {
      t = fl::abs(*(parsed_tensor->_tensor));
      t = fl::amax(t, axes, keep_dims);
    }
    auto axes_set = std::unordered_set<int>(axes.begin(), axes.end());
    auto base_shape = parsed_tensor->_tensor->shape().get();
    std::vector<fl::Dim> new_shape;
    for (size_t idx = 0; idx < base_shape.size(); ++idx) {
      if (axes_set.count(idx) || (axes_set.size() == 0)) {
        if (keep_dims) {
          new_shape.emplace_back(1);
        }
        continue;
      }
      new_shape.emplace_back(base_shape[idx]);
    }
    const auto& shape = fl::Shape(new_shape);
    t = fl::reshape(t, shape);
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value CountNonZero(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 3) {
    Napi::TypeError::New(info.Env(), "`countNonzero` expects exactly 3 args")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsObject()) {
    Napi::TypeError::New(
        info.Env(),
        "`countNonzero` expects 1st argument to be instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[1].IsArray()) {
    Napi::TypeError::New(
        info.Env(),
        "`countNonzero` expects 2nd argument to be typeof `number[]`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[2].IsBoolean()) {
    Napi::TypeError::New(
        info.Env(),
        "`countNonzero` expects 3rd argument to be typeof `boolean`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  bool keep_dims = info[2].As<Napi::Boolean>().Value();
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    auto axes = jsArrayArg<int>(info[1].As<Napi::Array>(), g_row_major,
                                parsed_tensor->_tensor->ndim(), env);
    fl::Tensor t;
    t = fl::countNonzero(*(parsed_tensor->_tensor), axes, keep_dims);
    auto axes_set = std::unordered_set<int>(axes.begin(), axes.end());
    auto base_shape = parsed_tensor->_tensor->shape().get();
    std::vector<fl::Dim> new_shape;
    for (size_t idx = 0; idx < base_shape.size(); ++idx) {
      if (axes_set.count(idx) || (axes_set.size() == 0)) {
        if (keep_dims) {
          new_shape.emplace_back(1);
        }
        continue;
      }
      new_shape.emplace_back(base_shape[idx]);
    }
    const auto& shape = fl::Shape(new_shape);
    t = fl::reshape(t, shape);
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value Any(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 3) {
    Napi::TypeError::New(info.Env(), "`any` expects exactly 3 args")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsObject()) {
    Napi::TypeError::New(info.Env(),
                         "`any` expects 1st argument to be instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[1].IsArray()) {
    Napi::TypeError::New(info.Env(),
                         "`any` expects 2nd argument to be typeof `number[]`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[2].IsBoolean()) {
    Napi::TypeError::New(info.Env(),
                         "`any` expects 3rd argument to be typeof `boolean`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  bool keep_dims = info[2].As<Napi::Boolean>().Value();
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    auto axes = jsArrayArg<int>(info[1].As<Napi::Array>(), g_row_major,
                                parsed_tensor->_tensor->ndim(), env);
    fl::Tensor t;
    t = fl::any(*(parsed_tensor->_tensor), axes, keep_dims);
    auto axes_set = std::unordered_set<int>(axes.begin(), axes.end());
    auto base_shape = parsed_tensor->_tensor->shape().get();
    std::vector<fl::Dim> new_shape;
    for (size_t idx = 0; idx < base_shape.size(); ++idx) {
      if (axes_set.count(idx) || (axes_set.size() == 0)) {
        if (keep_dims) {
          new_shape.emplace_back(1);
        }
        continue;
      }
      new_shape.emplace_back(base_shape[idx]);
    }
    const auto& shape = fl::Shape(new_shape);
    t = fl::reshape(t, shape);
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

static Napi::Value All(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 3) {
    Napi::TypeError::New(info.Env(), "`all` expects exactly 3 args")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsObject()) {
    Napi::TypeError::New(info.Env(),
                         "`all` expects 1st argument to be instanceof `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[1].IsArray()) {
    Napi::TypeError::New(info.Env(),
                         "`all` expects 2nd argument to be typeof `number[]`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[2].IsBoolean()) {
    Napi::TypeError::New(info.Env(),
                         "`all` expects 3rd argument to be typeof `boolean`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  bool keep_dims = info[2].As<Napi::Boolean>().Value();
  Napi::Object tensor_obj = info[0].As<Napi::Object>();
  if (tensor_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* parsed_tensor = Napi::ObjectWrap<Tensor>::Unwrap(tensor_obj);
    auto axes = jsArrayArg<int>(info[1].As<Napi::Array>(), g_row_major,
                                parsed_tensor->_tensor->ndim(), env);
    fl::Tensor t;
    t = fl::all(*(parsed_tensor->_tensor), axes, keep_dims);
    auto axes_set = std::unordered_set<int>(axes.begin(), axes.end());
    auto base_shape = parsed_tensor->_tensor->shape().get();
    std::vector<fl::Dim> new_shape;
    for (size_t idx = 0; idx < base_shape.size(); ++idx) {
      if (axes_set.count(idx) || (axes_set.size() == 0)) {
        if (keep_dims) {
          new_shape.emplace_back(1);
        }
        continue;
      }
      new_shape.emplace_back(base_shape[idx]);
    }
    const auto& shape = fl::Shape(new_shape);
    t = fl::reshape(t, shape);
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

// tensor class constructor
Tensor::Tensor(const Napi::CallbackInfo& info) : ObjectWrap(info) {
  Napi::Env env = info.Env();

  // throw error if number of args passed to constructor is not 1
  if (info.Length() != 1) {
    Napi::TypeError::New(env,
                         "Invalid arg count when constructing Tensor "
                         "(constructor expects exactly 1 arg)...")
        .ThrowAsJavaScriptException();
    return;
  }

  if (info[0].IsExternal()) {
    auto tensor = info[0].As<Napi::External<fl::Tensor>>();
    this->_tensor = tensor.Data();
  }

  // TODO: handle `arg instanceof Tensor`
  if (info[0].IsObject()) {
    Napi::Object obj = info[0].As<Napi::Object>();
    if (obj.InstanceOf(Tensor::constructor->Value())) {
      Tensor* t = Napi::ObjectWrap<Tensor>::Unwrap(obj);
      this->_underlying = t->_underlying;
      this->_tensor = t->_tensor;

      // TODO: finish handling this case
      return;
    }
  }

  // TODO: handle `typeof arg === 'string
  if (info[0].IsString()) {
    Napi::String str = info[0].As<Napi::String>();
    std::string filename = str.Utf8Value();
    fl::Tensor* t = load(filename, env);
    this->_tensor = t;
    return;
  }

  // handle if arg is a TypedArray
  if (info[0].IsTypedArray()) {
    Napi::TypedArray underlying = info[0].As<Napi::TypedArray>();
    this->_underlying = underlying;
    int64_t length = static_cast<int64_t>(underlying.ElementLength());
    napi_typedarray_type arrayType = underlying.TypedArrayType();
    switch (arrayType) {
      case napi_float32_array: {
        Napi::TypedArrayOf<float> float_array =
            underlying.As<Napi::TypedArrayOf<float>>();
        float* ptr = float_array.Data();
        auto* t = new fl::Tensor(
            fl::Tensor::fromBuffer({length}, ptr, fl::MemoryLocation::Host));
        g_bytes_used += t->bytes();
        this->_tensor = t;
        return;
      }
      case napi_float64_array: {
        Napi::TypedArrayOf<double> double_array =
            underlying.As<Napi::TypedArrayOf<double>>();
        double* ptr = double_array.Data();
        auto* t = new fl::Tensor(
            fl::Tensor::fromBuffer({length}, ptr, fl::MemoryLocation::Host));
        g_bytes_used += t->bytes();
        this->_tensor = t;
        return;
      }
      case napi_int8_array: {
        Napi::TypedArrayOf<int8_t> int8_array =
            underlying.As<Napi::TypedArrayOf<int8_t>>();
        char* ptr = reinterpret_cast<char*>(int8_array.Data());
        auto* t = new fl::Tensor(
            fl::Tensor::fromBuffer({length}, ptr, fl::MemoryLocation::Host));
        g_bytes_used += t->bytes();
        this->_tensor = t;
        return;
      }
      case napi_uint8_array: {
        Napi::TypedArrayOf<uint8_t> uint8_array =
            underlying.As<Napi::TypedArrayOf<uint8_t>>();
        uint8_t* ptr = uint8_array.Data();
        auto* t = new fl::Tensor(
            fl::Tensor::fromBuffer({length}, ptr, fl::MemoryLocation::Host));
        g_bytes_used += t->bytes();
        this->_tensor = t;
        return;
      }
      case napi_int16_array: {
        Napi::TypedArrayOf<int16_t> int16_array =
            underlying.As<Napi::TypedArrayOf<int16_t>>();
        int16_t* ptr = int16_array.Data();
        auto* t = new fl::Tensor(
            fl::Tensor::fromBuffer({length}, ptr, fl::MemoryLocation::Host));
        g_bytes_used += t->bytes();
        this->_tensor = t;
        return;
      }
      case napi_uint16_array: {
        Napi::TypedArrayOf<uint16_t> uint16_array =
            underlying.As<Napi::TypedArrayOf<uint16_t>>();
        uint16_t* ptr = uint16_array.Data();
        auto* t = new fl::Tensor(
            fl::Tensor::fromBuffer({length}, ptr, fl::MemoryLocation::Host));
        g_bytes_used += t->bytes();
        this->_tensor = t;
        return;
      }
      case napi_int32_array: {
        Napi::TypedArrayOf<int32_t> int32_array =
            underlying.As<Napi::TypedArrayOf<int32_t>>();
        int32_t* ptr = int32_array.Data();
        auto* t = new fl::Tensor(
            fl::Tensor::fromBuffer({length}, ptr, fl::MemoryLocation::Host));
        g_bytes_used += t->bytes();
        this->_tensor = t;
        return;
      }
      case napi_uint32_array: {
        Napi::TypedArrayOf<uint32_t> uint32_array =
            underlying.As<Napi::TypedArrayOf<uint32_t>>();
        uint32_t* ptr = uint32_array.Data();
        auto* t = new fl::Tensor(
            fl::Tensor::fromBuffer({length}, ptr, fl::MemoryLocation::Host));
        g_bytes_used += t->bytes();
        this->_tensor = t;
        return;
      }
      case napi_bigint64_array: {
        Napi::TypedArrayOf<int64_t> bigint64_array =
            underlying.As<Napi::TypedArrayOf<int64_t>>();
        int64_t* ptr = bigint64_array.Data();
        auto* t = new fl::Tensor(
            fl::Tensor::fromBuffer({length}, ptr, fl::MemoryLocation::Host));
        g_bytes_used += t->bytes();
        this->_tensor = t;
        return;
      }
      case napi_biguint64_array: {
        Napi::TypedArrayOf<uint64_t> biguint64_array =
            underlying.As<Napi::TypedArrayOf<uint64_t>>();
        uint64_t* ptr = biguint64_array.Data();
        auto* t = new fl::Tensor(
            fl::Tensor::fromBuffer({length}, ptr, fl::MemoryLocation::Host));
        g_bytes_used += t->bytes();
        this->_tensor = t;
        return;
      }
      default: {
        Napi::TypeError::New(env, "Unhandled TypedArray type")
            .ThrowAsJavaScriptException();
        return;
      }
    }
  }

  // TODO: handle `typeof arg === 'number'`
  if (info[0].IsNumber()) {
    const size_t length = 1;
    Napi::TypedArrayOf<int64_t> arr =
        Napi::TypedArrayOf<int64_t>::New(env, length, napi_bigint64_array);
    arr[0] = info[0].As<Napi::Number>().Int64Value();
    auto* ptr = arr.Data();
    auto shape = arrayArg<long long>(ptr, 1, g_row_major, false);
    auto* t = new fl::Tensor(fl::Shape(shape));
    g_bytes_used += t->bytes();
    this->_tensor = t;
    return;
  }
}

// tensor class methods
Napi::Value Tensor::Elements(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  return Napi::Number::New(env, this->_tensor->elements());
}

Napi::Value Tensor::NDim(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  return Napi::Number::New(env, this->_tensor->ndim());
}

Napi::Value Tensor::Dtype(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  return Napi::Number::New(env, static_cast<int>(this->_tensor->type()));
}

Napi::Value Tensor::Bytes(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  return Napi::BigInt::New(env, static_cast<uint64_t>(this->_tensor->bytes()));
}

void Tensor::Eval(const Napi::CallbackInfo& info) {
  fl::eval(*(this->_tensor));
}

void Tensor::Save(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1) {
    Napi::TypeError::New(
        env, "Tensor method `save` expects exactly 1 arg (string)... ")
        .ThrowAsJavaScriptException();
    return;
  }
  if (!info[0].IsString()) {
    Napi::TypeError::New(
        env,
        "Tensor method `save` requires arg `filename` of type `string`... ")
        .ThrowAsJavaScriptException();
    return;
  }
  Napi::String str = info[0].As<Napi::String>();
  std::string filename = str.Utf8Value();
  fl::save(filename, *(this->_tensor));
  return;
}

Napi::Value Tensor::Shape64(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  const size_t length = static_cast<const size_t>(this->_tensor->ndim());
  Napi::TypedArrayOf<int64_t> arr =
      Napi::TypedArrayOf<int64_t>::New(env, length, napi_bigint64_array);
  const int out_len = static_cast<int>(length);
  for (auto i = 0; i < out_len; ++i) {
    const auto idx = g_row_major ? out_len - i - 1 : i;
    arr[i] = static_cast<long long>(this->_tensor->shape()[idx]);
  }
  return arr;
}

Napi::Value Tensor::Shape(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  Napi::Array arr = Napi::Array::New(env);
  Napi::TypedArrayOf<int64_t> shape =
      this->Shape64(info).As<Napi::TypedArrayOf<int64_t>>();
  const int out_len = static_cast<int>(shape.ElementLength());
  for (auto i = 0; i < out_len; ++i) {
    arr[i] = Napi::Number::New(env, static_cast<double>(shape[i]));
  }
  return arr;
}

Napi::Value Tensor::ToString(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  uintptr_t ptr = reinterpret_cast<uintptr_t>(this->_tensor);
  std::string strVal = "Tensor[id=" + std::to_string(ptr) + "]";
  return Napi::String::New(env, strVal);
}

Napi::Value Tensor::ToFloat32Array(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  size_t elemLen = static_cast<size_t>(this->_tensor->elements());
  size_t byteLen = elemLen * sizeof(float);
  void* ptr = this->_tensor->astype(fl::dtype::f32).host<float>();
  Napi::ArrayBuffer buff = Napi::ArrayBuffer::New(env, ptr, byteLen);
  Napi::TypedArrayOf<float> out =
      Napi::TypedArrayOf<float>::New(env, elemLen, buff, 0, napi_float32_array);
  return out;
}

Napi::Value Tensor::ToFloat64Array(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  size_t elemLen = static_cast<size_t>(this->_tensor->elements());
  size_t byteLen = elemLen * sizeof(double);
  void* ptr = this->_tensor->astype(fl::dtype::f64).host<double>();
  Napi::ArrayBuffer buff = Napi::ArrayBuffer::New(env, ptr, byteLen);
  Napi::TypedArrayOf<double> out = Napi::TypedArrayOf<double>::New(
      env, elemLen, buff, 0, napi_float64_array);
  return out;
}

Napi::Value Tensor::ToBoolInt8Array(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  size_t elemLen = static_cast<size_t>(this->_tensor->elements());
  size_t byteLen = elemLen * sizeof(int8_t);
  void* ptr = this->_tensor->astype(fl::dtype::b8).host<int>();
  Napi::ArrayBuffer buff = Napi::ArrayBuffer::New(env, ptr, byteLen);
  Napi::TypedArrayOf<int8_t> out =
      Napi::TypedArrayOf<int8_t>::New(env, elemLen, buff, 0, napi_int8_array);
  return out;
}

Napi::Value Tensor::ToInt16Array(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  size_t elemLen = static_cast<size_t>(this->_tensor->elements());
  size_t byteLen = elemLen * sizeof(int16_t);
  void* ptr = this->_tensor->astype(fl::dtype::s16).host<int>();
  Napi::ArrayBuffer buff = Napi::ArrayBuffer::New(env, ptr, byteLen);
  Napi::TypedArrayOf<int16_t> out =
      Napi::TypedArrayOf<int16_t>::New(env, elemLen, buff, 0, napi_int16_array);
  return out;
}

Napi::Value Tensor::ToInt32Array(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  size_t elemLen = static_cast<size_t>(this->_tensor->elements());
  size_t byteLen = elemLen * sizeof(int32_t);
  void* ptr = this->_tensor->astype(fl::dtype::s32).host<int>();
  Napi::ArrayBuffer buff = Napi::ArrayBuffer::New(env, ptr, byteLen);
  Napi::TypedArrayOf<int32_t> out =
      Napi::TypedArrayOf<int32_t>::New(env, elemLen, buff, 0, napi_int32_array);
  return out;
}

Napi::Value Tensor::ToBigInt64Array(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  size_t elemLen = static_cast<size_t>(this->_tensor->elements());
  size_t byteLen = elemLen * sizeof(int64_t);
  void* ptr = this->_tensor->astype(fl::dtype::s64).host<int>();
  Napi::ArrayBuffer buff = Napi::ArrayBuffer::New(env, ptr, byteLen);
  Napi::TypedArrayOf<int64_t> out = Napi::TypedArrayOf<int64_t>::New(
      env, elemLen, buff, 0, napi_bigint64_array);
  return out;
}

Napi::Value Tensor::ToUint8Array(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  size_t elemLen = static_cast<size_t>(this->_tensor->elements());
  size_t byteLen = elemLen * sizeof(uint8_t);
  void* ptr = this->_tensor->astype(fl::dtype::u8).host<unsigned>();
  Napi::ArrayBuffer buff = Napi::ArrayBuffer::New(env, ptr, byteLen);
  Napi::TypedArrayOf<uint8_t> out =
      Napi::TypedArrayOf<uint8_t>::New(env, elemLen, buff, 0, napi_uint8_array);
  return out;
}

Napi::Value Tensor::ToUint16Array(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  size_t elemLen = static_cast<size_t>(this->_tensor->elements());
  size_t byteLen = elemLen * sizeof(uint16_t);
  void* ptr = this->_tensor->astype(fl::dtype::u16).host<unsigned>();
  Napi::ArrayBuffer buff = Napi::ArrayBuffer::New(env, ptr, byteLen);
  Napi::TypedArrayOf<uint16_t> out = Napi::TypedArrayOf<uint16_t>::New(
      env, elemLen, buff, 0, napi_uint16_array);
  return out;
}

Napi::Value Tensor::ToUint32Array(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  size_t elemLen = static_cast<size_t>(this->_tensor->elements());
  size_t byteLen = elemLen * sizeof(uint32_t);
  void* ptr = this->_tensor->astype(fl::dtype::u32).host<unsigned>();
  Napi::ArrayBuffer buff = Napi::ArrayBuffer::New(env, ptr, byteLen);
  Napi::TypedArrayOf<uint32_t> out = Napi::TypedArrayOf<uint32_t>::New(
      env, elemLen, buff, 0, napi_uint32_array);
  return out;
}

Napi::Value Tensor::ToBigUint64Array(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  size_t elemLen = static_cast<size_t>(this->_tensor->elements());
  size_t byteLen = elemLen * sizeof(uint64_t);
  void* ptr = this->_tensor->astype(fl::dtype::u64).host<unsigned>();
  Napi::ArrayBuffer buff = Napi::ArrayBuffer::New(env, ptr, byteLen);
  Napi::TypedArrayOf<uint64_t> out = Napi::TypedArrayOf<uint64_t>::New(
      env, elemLen, buff, 0, napi_biguint64_array);
  return out;
}

void Tensor::Update(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1) {
    Napi::TypeError::New(env,
                         "Tensor method `update` expects exactly 1 arg "
                         "(instanceof `Tensor`)... ")
        .ThrowAsJavaScriptException();
    return;
  }
  if (!info[0].ToObject().InstanceOf(Tensor::GetClass(env))) {
    Napi::TypeError::New(env,
                         "Tensor method `update` requires arg `tensor` "
                         "instanceof `Tensor`... ")
        .ThrowAsJavaScriptException();
    return;
  }
  Tensor* tensor = Napi::ObjectWrap<Tensor>::Unwrap(info[0].ToObject());
  this->_underlying = tensor->_underlying;
  this->_tensor = tensor->_tensor;
  fl::eval(*(this->_tensor));
}

void Tensor::Dispose(const Napi::CallbackInfo& info) {
  auto& tensor = *reinterpret_cast<fl::Tensor*>(this->_tensor);
  g_bytes_used -= tensor.bytes();
  fl::detail::releaseAdapterUnsafe(tensor);
}

Napi::Value Tensor::Reshape(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1 || !info[0].IsArray()) {
    Napi::TypeError::New(info.Env(),
                         "Tensor method `reshape` expects exactly 1 arg; "
                         "(expected type `number[]`)...")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  std::vector<long long> shape =
      jsArrayArg<long long>(info[0].As<Napi::Array>(), g_row_major, false, env);
  fl::Tensor t;
  t = fl::reshape(*(this->_tensor), fl::Shape(shape));
  g_bytes_used += t.bytes();
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

Napi::Value Tensor::AsType(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1 || !info[0].IsNumber()) {
    Napi::TypeError::New(info.Env(),
                         "Tensor method `astype` expects exactly 1 arg; "
                         "(expected type `number[]`)...")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto dtype = static_cast<fl::dtype>(info[0].As<Napi::Number>().Int32Value());
  auto t = this->_tensor->astype(dtype);
  g_bytes_used += t.bytes();
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

Napi::Value Tensor::Transpose(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1 || !info[0].IsArray()) {
    Napi::TypeError::New(info.Env(),
                         "Tensor method `transpose` expects exactly 1 arg; "
                         "(expected type `number[]`)...")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  Napi::Array axes_arg = info[0].As<Napi::Array>();

  std::vector<long long> axes =
      jsArrayArg<long long>(axes_arg, g_row_major, this->_tensor->ndim(), env);

  fl::Tensor t;
  t = fl::transpose(*(this->_tensor), fl::Shape(axes));
  g_bytes_used += t.bytes();
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

Napi::Value Tensor::Tile(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1 || !info[0].IsArray()) {
    Napi::TypeError::New(info.Env(),
                         "Tensor method `tile` expects exactly 1 arg; "
                         "(expected type `number[]`)...")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  Napi::Array shape_arg = info[0].As<Napi::Array>();
  std::vector<long long> shape =
      jsArrayArg<long long>(shape_arg, g_row_major, false, env);
  fl::Tensor t;
  t = fl::tile(*(this->_tensor), fl::Shape(shape));
  g_bytes_used += t.bytes();
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

Napi::Value Tensor::NonZero(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  t = fl::nonzero(*(this->_tensor));
  g_bytes_used += t.bytes();
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

Napi::Value Tensor::Negative(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  t = fl::negative(*(this->_tensor));
  g_bytes_used += t.bytes();
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

Napi::Value Tensor::LogicalNot(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  t = fl::logicalNot(*(this->_tensor));
  g_bytes_used += t.bytes();
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

Napi::Value Tensor::Exp(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  t = fl::exp(*(this->_tensor));
  g_bytes_used += t.bytes();
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

Napi::Value Tensor::Log(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  t = fl::log(*(this->_tensor));
  g_bytes_used += t.bytes();
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

Napi::Value Tensor::Log1p(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  t = fl::log1p(*(this->_tensor));
  g_bytes_used += t.bytes();
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

Napi::Value Tensor::Sin(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  t = fl::sin(*(this->_tensor));
  g_bytes_used += t.bytes();
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

Napi::Value Tensor::Cos(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  t = fl::cos(*(this->_tensor));
  g_bytes_used += t.bytes();
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

Napi::Value Tensor::Sqrt(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  t = fl::sqrt(*(this->_tensor));
  g_bytes_used += t.bytes();
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

Napi::Value Tensor::Tanh(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  t = fl::tanh(*(this->_tensor));
  g_bytes_used += t.bytes();
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

Napi::Value Tensor::Floor(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  t = fl::floor(*(this->_tensor));
  g_bytes_used += t.bytes();
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

Napi::Value Tensor::Ceil(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  t = fl::ceil(*(this->_tensor));
  g_bytes_used += t.bytes();
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

Napi::Value Tensor::Rint(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  t = fl::rint(*(this->_tensor));
  g_bytes_used += t.bytes();
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

Napi::Value Tensor::Absolute(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  t = fl::absolute(*(this->_tensor));
  g_bytes_used += t.bytes();
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

Napi::Value Tensor::Sigmoid(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  t = fl::sigmoid(*(this->_tensor));
  g_bytes_used += t.bytes();
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

Napi::Value Tensor::Erf(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  t = fl::erf(*(this->_tensor));
  g_bytes_used += t.bytes();
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

Napi::Value Tensor::Flip(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1 || !info[0].IsNumber()) {
    Napi::TypeError::New(info.Env(),
                         "Tensor method `flip` expects exactly 1 arg, (arg "
                         "`dim` of type `number`)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  uint32_t dim = info[0].As<Napi::Number>().Uint32Value();
  fl::Tensor t;
  t = fl::flip(*(this->_tensor), dim);
  g_bytes_used += t.bytes();
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

Napi::Value Tensor::Clip(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  if (info.Length() != 2 || (!info[0].IsObject() || !info[1].IsObject())) {
    Napi::TypeError::New(
        info.Env(),
        "Tensor method `clip` expects exactly 2 args, each of type `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object low_obj = info[0].As<Napi::Object>();
  Napi::Object high_obj = info[1].As<Napi::Object>();
  if (low_obj.InstanceOf(Tensor::constructor->Value()) &&
      high_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* low_tensor = Napi::ObjectWrap<Tensor>::Unwrap(low_obj);
    Tensor* high_tensor = Napi::ObjectWrap<Tensor>::Unwrap(high_obj);
    t = fl::clip(*(this->_tensor), *(low_tensor->_tensor),
                 *(high_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

Napi::Value Tensor::Roll(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  if (info.Length() != 2 || (!info[0].IsNumber() || !info[1].IsNumber())) {
    Napi::TypeError::New(
        info.Env(),
        "Tensor method `roll` expects exactly 2 args, each of type `number`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  int shift = static_cast<int>(info[0].As<Napi::Number>().Int64Value());
  int32_t axis = info[1].As<Napi::Number>().Int32Value();
  auto used_axis = axisArg(axis, g_row_major, this->_tensor->ndim());
  t = fl::roll(*(this->_tensor), shift, used_axis);
  g_bytes_used += t.bytes();
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

Napi::Value Tensor::IsNaN(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  t = fl::isnan(*(this->_tensor));
  g_bytes_used += t.bytes();
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

Napi::Value Tensor::IsInf(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  t = fl::isinf(*(this->_tensor));
  g_bytes_used += t.bytes();
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

Napi::Value Tensor::Sign(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  t = fl::sign(*(this->_tensor));
  g_bytes_used += t.bytes();
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

Napi::Value Tensor::Tril(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  if (g_row_major) {
    t = fl::triu(*(this->_tensor));
  } else {
    t = fl::tril(*(this->_tensor));
  }
  g_bytes_used += t.bytes();
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

Napi::Value Tensor::Triu(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  if (g_row_major) {
    t = fl::tril(*(this->_tensor));
  } else {
    t = fl::triu(*(this->_tensor));
  }
  g_bytes_used += t.bytes();
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

Napi::Value Tensor::Where(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  if (info.Length() != 2 || (!info[0].IsObject() || !info[1].IsObject())) {
    Napi::TypeError::New(
        info.Env(),
        "Tensor method `where` expects exactly 2 args, each of type `Tensor`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object x_obj = info[0].As<Napi::Object>();
  Napi::Object y_obj = info[1].As<Napi::Object>();
  if (x_obj.InstanceOf(Tensor::constructor->Value()) &&
      y_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* x_tensor = Napi::ObjectWrap<Tensor>::Unwrap(x_obj);
    Tensor* y_tensor = Napi::ObjectWrap<Tensor>::Unwrap(y_obj);
    t = fl::where(this->_tensor->astype(fl::dtype::b8), *(x_tensor->_tensor),
                  *(y_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

Napi::Value Tensor::Sort(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  if (info.Length() != 1 || !info[0].IsNumber()) {
    Napi::TypeError::New(
        info.Env(),
        "Tensor method `sort` expects exactly 1 arg (`dim` of type `number`)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  int dim = info[0].As<Napi::Number>().Uint32Value();
  t = fl::sort(*(this->_tensor), dim);
  g_bytes_used += t.bytes();
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

Napi::Value Tensor::Add(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  if (info.Length() != 1 || !info[0].IsObject()) {
    Napi::TypeError::New(
        info.Env(),
        "Tensor method `add` expects exactly 1 arg (`other` of type `Tensor`)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object other_obj = info[0].As<Napi::Object>();
  if (other_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* other_tensor = Napi::ObjectWrap<Tensor>::Unwrap(other_obj);
    t = fl::add(*(this->_tensor), *(other_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

Napi::Value Tensor::Sub(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  if (info.Length() != 1 || !info[0].IsObject()) {
    Napi::TypeError::New(
        info.Env(),
        "Tensor method `add` expects exactly 1 arg (`other` of type `Tensor`)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object other_obj = info[0].As<Napi::Object>();
  if (other_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* other_tensor = Napi::ObjectWrap<Tensor>::Unwrap(other_obj);
    t = fl::sub(*(this->_tensor), *(other_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

Napi::Value Tensor::Mul(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  if (info.Length() != 1 || !info[0].IsObject()) {
    Napi::TypeError::New(
        info.Env(),
        "Tensor method `add` expects exactly 1 arg (`other` of type `Tensor`)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object other_obj = info[0].As<Napi::Object>();
  if (other_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* other_tensor = Napi::ObjectWrap<Tensor>::Unwrap(other_obj);
    t = fl::mul(*(this->_tensor), *(other_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

Napi::Value Tensor::Div(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  if (info.Length() != 1 || !info[0].IsObject()) {
    Napi::TypeError::New(
        info.Env(),
        "Tensor method `add` expects exactly 1 arg (`other` of type `Tensor`)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object other_obj = info[0].As<Napi::Object>();
  if (other_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* other_tensor = Napi::ObjectWrap<Tensor>::Unwrap(other_obj);
    t = fl::div(*(this->_tensor), *(other_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

Napi::Value Tensor::Eq(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  if (info.Length() != 1 || !info[0].IsObject()) {
    Napi::TypeError::New(
        info.Env(),
        "Tensor method `add` expects exactly 1 arg (`other` of type `Tensor`)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object other_obj = info[0].As<Napi::Object>();
  if (other_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* other_tensor = Napi::ObjectWrap<Tensor>::Unwrap(other_obj);
    t = fl::eq(*(this->_tensor), *(other_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

Napi::Value Tensor::Neq(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  if (info.Length() != 1 || !info[0].IsObject()) {
    Napi::TypeError::New(
        info.Env(),
        "Tensor method `add` expects exactly 1 arg (`other` of type `Tensor`)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object other_obj = info[0].As<Napi::Object>();
  if (other_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* other_tensor = Napi::ObjectWrap<Tensor>::Unwrap(other_obj);
    t = fl::neq(*(this->_tensor), *(other_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

Napi::Value Tensor::LessThan(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  if (info.Length() != 1 || !info[0].IsObject()) {
    Napi::TypeError::New(info.Env(),
                         "Tensor method `lessThan (lt)` expects exactly 1 arg "
                         "(`other` of type `Tensor`)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object other_obj = info[0].As<Napi::Object>();
  if (other_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* other_tensor = Napi::ObjectWrap<Tensor>::Unwrap(other_obj);
    t = fl::lessThan(*(this->_tensor), *(other_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

Napi::Value Tensor::LessThanEqual(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  if (info.Length() != 1 || !info[0].IsObject()) {
    Napi::TypeError::New(info.Env(),
                         "Tensor method `lessThanEqual` expects exactly 1 arg "
                         "(`other` of type `Tensor`)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object other_obj = info[0].As<Napi::Object>();
  if (other_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* other_tensor = Napi::ObjectWrap<Tensor>::Unwrap(other_obj);
    t = fl::lessThanEqual(*(this->_tensor), *(other_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

Napi::Value Tensor::GreaterThan(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  if (info.Length() != 1 || !info[0].IsObject()) {
    Napi::TypeError::New(info.Env(),
                         "Tensor method `greaterThan` expects exactly 1 arg "
                         "(`other` of type `Tensor`)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object other_obj = info[0].As<Napi::Object>();
  if (other_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* other_tensor = Napi::ObjectWrap<Tensor>::Unwrap(other_obj);
    t = fl::greaterThan(*(this->_tensor), *(other_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

Napi::Value Tensor::GreaterThanEqual(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  if (info.Length() != 1 || !info[0].IsObject()) {
    Napi::TypeError::New(
        info.Env(),
        "Tensor method `greaterThanEqual` expects exactly 1 arg "
        "(`other` of type `Tensor`)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object other_obj = info[0].As<Napi::Object>();
  if (other_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* other_tensor = Napi::ObjectWrap<Tensor>::Unwrap(other_obj);
    t = fl::greaterThanEqual(*(this->_tensor), *(other_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

Napi::Value Tensor::LogicalOr(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  if (info.Length() != 1 || !info[0].IsObject()) {
    Napi::TypeError::New(info.Env(),
                         "Tensor method `logicalOr` expects exactly 1 arg "
                         "(`other` of type `Tensor`)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object other_obj = info[0].As<Napi::Object>();
  if (other_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* other_tensor = Napi::ObjectWrap<Tensor>::Unwrap(other_obj);
    t = fl::logicalOr(*(this->_tensor), *(other_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

Napi::Value Tensor::LogicalAnd(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  if (info.Length() != 1 || !info[0].IsObject()) {
    Napi::TypeError::New(info.Env(),
                         "Tensor method `logicalAnd` expects exactly 1 arg "
                         "(`other` of type `Tensor`)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object other_obj = info[0].As<Napi::Object>();
  if (other_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* other_tensor = Napi::ObjectWrap<Tensor>::Unwrap(other_obj);
    t = fl::logicalAnd(*(this->_tensor), *(other_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

Napi::Value Tensor::Mod(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  if (info.Length() != 1 || !info[0].IsObject()) {
    Napi::TypeError::New(info.Env(),
                         "Tensor method `mod` expects exactly 1 arg "
                         "(`other` of type `Tensor`)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object other_obj = info[0].As<Napi::Object>();
  if (other_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* other_tensor = Napi::ObjectWrap<Tensor>::Unwrap(other_obj);
    t = fl::mod(*(this->_tensor), *(other_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

Napi::Value Tensor::BitwiseAnd(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  if (info.Length() != 1 || !info[0].IsObject()) {
    Napi::TypeError::New(info.Env(),
                         "Tensor method `bitwiseAnd` expects exactly 1 arg "
                         "(`other` of type `Tensor`)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object other_obj = info[0].As<Napi::Object>();
  if (other_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* other_tensor = Napi::ObjectWrap<Tensor>::Unwrap(other_obj);
    t = fl::bitwiseAnd(*(this->_tensor), *(other_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

Napi::Value Tensor::BitwiseOr(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  if (info.Length() != 1 || !info[0].IsObject()) {
    Napi::TypeError::New(info.Env(),
                         "Tensor method `bitwiseAnd` expects exactly 1 arg "
                         "(`other` of type `Tensor`)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object other_obj = info[0].As<Napi::Object>();
  if (other_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* other_tensor = Napi::ObjectWrap<Tensor>::Unwrap(other_obj);
    t = fl::bitwiseOr(*(this->_tensor), *(other_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

Napi::Value Tensor::BitwiseXor(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  if (info.Length() != 1 || !info[0].IsObject()) {
    Napi::TypeError::New(info.Env(),
                         "Tensor method `bitwiseAnd` expects exactly 1 arg "
                         "(`other` of type `Tensor`)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object other_obj = info[0].As<Napi::Object>();
  if (other_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* other_tensor = Napi::ObjectWrap<Tensor>::Unwrap(other_obj);
    t = fl::bitwiseXor(*(this->_tensor), *(other_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

Napi::Value Tensor::LShift(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  if (info.Length() != 1 || !info[0].IsObject()) {
    Napi::TypeError::New(info.Env(),
                         "Tensor method `bitwiseAnd` expects exactly 1 arg "
                         "(`other` of type `Tensor`)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object other_obj = info[0].As<Napi::Object>();
  if (other_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* other_tensor = Napi::ObjectWrap<Tensor>::Unwrap(other_obj);
    t = fl::lShift(*(this->_tensor), *(other_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

Napi::Value Tensor::RShift(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  if (info.Length() != 1 || !info[0].IsObject()) {
    Napi::TypeError::New(info.Env(),
                         "Tensor method `bitwiseAnd` expects exactly 1 arg "
                         "(`other` of type `Tensor`)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object other_obj = info[0].As<Napi::Object>();
  if (other_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* other_tensor = Napi::ObjectWrap<Tensor>::Unwrap(other_obj);
    t = fl::rShift(*(this->_tensor), *(other_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

Napi::Value Tensor::Minimum(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  if (info.Length() != 1 || !info[0].IsObject()) {
    Napi::TypeError::New(info.Env(),
                         "Tensor method `bitwiseAnd` expects exactly 1 arg "
                         "(`other` of type `Tensor`)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object other_obj = info[0].As<Napi::Object>();
  if (other_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* other_tensor = Napi::ObjectWrap<Tensor>::Unwrap(other_obj);
    t = fl::minimum(*(this->_tensor), *(other_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

Napi::Value Tensor::Maximum(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  if (info.Length() != 1 || !info[0].IsObject()) {
    Napi::TypeError::New(info.Env(),
                         "Tensor method `bitwiseAnd` expects exactly 1 arg "
                         "(`other` of type `Tensor`)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object other_obj = info[0].As<Napi::Object>();
  if (other_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* other_tensor = Napi::ObjectWrap<Tensor>::Unwrap(other_obj);
    t = fl::maximum(*(this->_tensor), *(other_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

Napi::Value Tensor::Power(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  if (info.Length() != 1 || !info[0].IsObject()) {
    Napi::TypeError::New(info.Env(),
                         "Tensor method `bitwiseAnd` expects exactly 1 arg "
                         "(`other` of type `Tensor`)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object other_obj = info[0].As<Napi::Object>();
  if (other_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* other_tensor = Napi::ObjectWrap<Tensor>::Unwrap(other_obj);
    t = fl::power(*(this->_tensor), *(other_tensor->_tensor));
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

Napi::Value Tensor::MatMul(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  if (info.Length() != 1 || !info[0].IsObject()) {
    Napi::TypeError::New(info.Env(),
                         "Tensor method `bitwiseAnd` expects exactly 1 arg "
                         "(`other` of type `Tensor`)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Object other_obj = info[0].As<Napi::Object>();
  if (other_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* other_tensor = Napi::ObjectWrap<Tensor>::Unwrap(other_obj);
    fl::Tensor t;
    if (g_row_major) {
      t = fl::matmul(*(other_tensor->_tensor), *(this->_tensor));
    } else {
      t = fl::matmul(*(this->_tensor), *(other_tensor->_tensor));
    }
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

Napi::Value Tensor::Conv2d(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor t;
  if (info.Length() >= 1 || !info[0].IsObject()) {
    Napi::TypeError::New(info.Env(),
                         "Tensor method `conv2d` expects minimum 1 arg "
                         "(`weights` of type `Tensor`)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  int32_t sx, sy, px, py, dx, dy, groups;
  if (info.Length() >= 2 && info[1].IsNumber()) {
    sx = info[1].As<Napi::Number>().Int32Value();
  } else {
    sx = 1;
  }
  if (info.Length() >= 3 && info[2].IsNumber()) {
    sy = info[2].As<Napi::Number>().Int32Value();
  } else {
    sy = 1;
  }
  if (info.Length() >= 4 && info[3].IsNumber()) {
    px = info[3].As<Napi::Number>().Int32Value();
  } else {
    px = 0;
  }
  if (info.Length() >= 5 && info[4].IsNumber()) {
    py = info[4].As<Napi::Number>().Int32Value();
  } else {
    py = 0;
  }
  if (info.Length() >= 6 && info[5].IsNumber()) {
    dx = info[5].As<Napi::Number>().Int32Value();
  } else {
    dx = 1;
  }
  if (info.Length() >= 7 && info[6].IsNumber()) {
    dy = info[6].As<Napi::Number>().Int32Value();
  } else {
    dy = 1;
  }
  if (info.Length() >= 8 && info[7].IsNumber()) {
    groups = info[7].As<Napi::Number>().Int32Value();
  } else {
    groups = 1;
  }
  Napi::Object weights_obj = info[0].As<Napi::Object>();
  if (weights_obj.InstanceOf(Tensor::constructor->Value())) {
    Tensor* weights_tensor = Napi::ObjectWrap<Tensor>::Unwrap(weights_obj);
    fl::Tensor t;
    t = fl::conv2d(*(this->_tensor), *(weights_tensor->_tensor), sx, sy, px, py,
                   dx, dy, groups);
    g_bytes_used += t.bytes();
    auto* tensor = new fl::Tensor(t);
    auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
    Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
    return wrappedTensor;
  }
  return env.Null();
}

Napi::Value Tensor::AMin(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2) {
    Napi::TypeError::New(info.Env(),
                         "Tensor method `amin` expects exactly 2 args ")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsArray()) {
    Napi::TypeError::New(
        info.Env(),
        "Tensor method `amin` expects 1st arg to be typeof `number[]`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[1].IsBoolean()) {
    Napi::TypeError::New(
        info.Env(), "Tensor method `amin` expects 2nd arg to be typeof boolean")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Array axes_arg = info[0].As<Napi::Array>();
  bool keep_dims = info[1].As<Napi::Boolean>().Value();
  auto axes =
      jsArrayArg<int>(axes_arg, g_row_major, this->_tensor->ndim(), env);
  fl::Tensor t;
  t = fl::amin(*(this->_tensor), axes, keep_dims);
  auto axes_set = std::unordered_set<int>(axes.begin(), axes.end());
  auto base_shape = this->_tensor->shape().get();
  std::vector<fl::Dim> new_shape;
  for (size_t idx = 0; idx < base_shape.size(); ++idx) {
    if (axes_set.count(idx) || (axes_set.size() == 0)) {
      if (keep_dims) {
        new_shape.emplace_back(1);
      }
      continue;
    }
    new_shape.emplace_back(base_shape[idx]);
  }
  const auto& shape = fl::Shape(new_shape);
  t = fl::reshape(t, shape);
  g_bytes_used += t.bytes();
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

Napi::Value Tensor::AMax(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2) {
    Napi::TypeError::New(info.Env(),
                         "Tensor method `amax` expects exactly 2 args ")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsArray()) {
    Napi::TypeError::New(
        info.Env(),
        "Tensor method `amax` expects 1st arg to be typeof `number[]`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[1].IsBoolean()) {
    Napi::TypeError::New(
        info.Env(), "Tensor method `amax` expects 2nd arg to be typeof boolean")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Array axes_arg = info[0].As<Napi::Array>();
  bool keep_dims = info[1].As<Napi::Boolean>().Value();
  auto axes =
      jsArrayArg<int>(axes_arg, g_row_major, this->_tensor->ndim(), env);
  fl::Tensor t;
  t = fl::amax(*(this->_tensor), axes, keep_dims);
  auto axes_set = std::unordered_set<int>(axes.begin(), axes.end());
  auto base_shape = this->_tensor->shape().get();
  std::vector<fl::Dim> new_shape;
  for (size_t idx = 0; idx < base_shape.size(); ++idx) {
    if (axes_set.count(idx) || (axes_set.size() == 0)) {
      if (keep_dims) {
        new_shape.emplace_back(1);
      }
      continue;
    }
    new_shape.emplace_back(base_shape[idx]);
  }
  const auto& shape = fl::Shape(new_shape);
  t = fl::reshape(t, shape);
  g_bytes_used += t.bytes();
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

Napi::Value Tensor::ArgMin(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2) {
    Napi::TypeError::New(info.Env(),
                         "Tensor method `argmin` expects exactly 2 args ")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsNumber()) {
    Napi::TypeError::New(
        info.Env(),
        "Tensor method `argmin` expects 1st arg to be typeof `number`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[1].IsBoolean()) {
    Napi::TypeError::New(
        info.Env(),
        "Tensor method `argmin` expects 2nd arg to be typeof boolean")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  int32_t axis = info[0].As<Napi::Number>().Int32Value();
  bool keep_dims = info[1].As<Napi::Boolean>().Value();
  auto used_axis = axisArg(axis, g_row_major, this->_tensor->ndim());
  fl::Tensor t;
  t = fl::argmin(*(this->_tensor), used_axis, keep_dims);
  auto axes_set = std::unordered_set<int>{static_cast<int>(used_axis)};
  auto base_shape = this->_tensor->shape().get();
  std::vector<fl::Dim> new_shape;
  for (size_t idx = 0; idx < base_shape.size(); ++idx) {
    if (axes_set.count(idx) || (axes_set.size() == 0)) {
      if (keep_dims) {
        new_shape.emplace_back(1);
      }
      continue;
    }
    new_shape.emplace_back(base_shape[idx]);
  }
  const auto& shape = fl::Shape(new_shape);
  t = fl::reshape(t, shape);
  g_bytes_used += t.bytes();
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

Napi::Value Tensor::ArgMax(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2) {
    Napi::TypeError::New(info.Env(),
                         "Tensor method `argmax` expects exactly 2 args ")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsNumber()) {
    Napi::TypeError::New(
        info.Env(),
        "Tensor method `argmax` expects 1st arg to be typeof `number`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[1].IsBoolean()) {
    Napi::TypeError::New(
        info.Env(),
        "Tensor method `argmax` expects 2nd arg to be typeof boolean")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  int32_t axis = info[0].As<Napi::Number>().Int32Value();
  bool keep_dims = info[1].As<Napi::Boolean>().Value();
  auto used_axis = axisArg(axis, g_row_major, this->_tensor->ndim());
  fl::Tensor t;
  t = fl::argmax(*(this->_tensor), used_axis, keep_dims);
  auto axes_set = std::unordered_set<int>{static_cast<int>(used_axis)};
  auto base_shape = this->_tensor->shape().get();
  std::vector<fl::Dim> new_shape;
  for (size_t idx = 0; idx < base_shape.size(); ++idx) {
    if (axes_set.count(idx) || (axes_set.size() == 0)) {
      if (keep_dims) {
        new_shape.emplace_back(1);
      }
      continue;
    }
    new_shape.emplace_back(base_shape[idx]);
  }
  const auto& shape = fl::Shape(new_shape);
  t = fl::reshape(t, shape);
  g_bytes_used += t.bytes();
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

Napi::Value Tensor::Sum(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2) {
    Napi::TypeError::New(info.Env(),
                         "Tensor method `sum` expects exactly 2 args ")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsArray()) {
    Napi::TypeError::New(
        info.Env(),
        "Tensor method `sum` expects 1st arg to be typeof `number[]`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[1].IsBoolean()) {
    Napi::TypeError::New(
        info.Env(), "Tensor method `sum` expects 2nd arg to be typeof boolean")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Array axes_arg = info[0].As<Napi::Array>();
  bool keep_dims = info[1].As<Napi::Boolean>().Value();
  auto axes =
      jsArrayArg<int>(axes_arg, g_row_major, this->_tensor->ndim(), env);
  fl::Tensor t;
  t = fl::sum(*(this->_tensor), axes, keep_dims);
  auto axes_set = std::unordered_set<int>(axes.begin(), axes.end());
  auto base_shape = this->_tensor->shape().get();
  std::vector<fl::Dim> new_shape;
  for (size_t idx = 0; idx < base_shape.size(); ++idx) {
    if (axes_set.count(idx) || (axes_set.size() == 0)) {
      if (keep_dims) {
        new_shape.emplace_back(1);
      }
      continue;
    }
    new_shape.emplace_back(base_shape[idx]);
  }
  const auto& shape = fl::Shape(new_shape);
  t = fl::reshape(t, shape);
  g_bytes_used += t.bytes();
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

Napi::Value Tensor::CumSum(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1 || !info[0].IsNumber()) {
    Napi::TypeError::New(info.Env(),
                         "Tensor method `cumsum` expects exactly 1 arg, (arg "
                         "`axis` of type `number`)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  uint32_t axis = info[0].As<Napi::Number>().Uint32Value();
  auto used_axis = axisArg(axis, g_row_major, this->_tensor->ndim());
  fl::Tensor t;
  t = fl::cumsum(*(this->_tensor), used_axis);
  g_bytes_used += t.bytes();
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

Napi::Value Tensor::Mean(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2) {
    Napi::TypeError::New(info.Env(),
                         "Tensor method `mean` expects exactly 2 args ")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsArray()) {
    Napi::TypeError::New(
        info.Env(),
        "Tensor method `mean` expects 1st arg to be typeof `number[]`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[1].IsBoolean()) {
    Napi::TypeError::New(
        info.Env(), "Tensor method `mean` expects 2nd arg to be typeof boolean")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Array axes_arg = info[0].As<Napi::Array>();
  bool keep_dims = info[1].As<Napi::Boolean>().Value();
  auto axes =
      jsArrayArg<int>(axes_arg, g_row_major, this->_tensor->ndim(), env);
  fl::Tensor t;
  t = fl::mean(*(this->_tensor), axes, keep_dims);
  auto axes_set = std::unordered_set<int>(axes.begin(), axes.end());
  auto base_shape = this->_tensor->shape().get();
  std::vector<fl::Dim> new_shape;
  for (size_t idx = 0; idx < base_shape.size(); ++idx) {
    if (axes_set.count(idx) || (axes_set.size() == 0)) {
      if (keep_dims) {
        new_shape.emplace_back(1);
      }
      continue;
    }
    new_shape.emplace_back(base_shape[idx]);
  }
  const auto& shape = fl::Shape(new_shape);
  t = fl::reshape(t, shape);
  g_bytes_used += t.bytes();
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

Napi::Value Tensor::Median(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2) {
    Napi::TypeError::New(info.Env(),
                         "Tensor method `median` expects exactly 2 args ")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsArray()) {
    Napi::TypeError::New(
        info.Env(),
        "Tensor method `median` expects 1st arg to be typeof `number[]`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[1].IsBoolean()) {
    Napi::TypeError::New(
        info.Env(),
        "Tensor method `median` expects 2nd arg to be typeof boolean")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Array axes_arg = info[0].As<Napi::Array>();
  bool keep_dims = info[1].As<Napi::Boolean>().Value();
  auto axes =
      jsArrayArg<int>(axes_arg, g_row_major, this->_tensor->ndim(), env);
  fl::Tensor t;
  t = fl::median(*(this->_tensor), axes, keep_dims);
  auto axes_set = std::unordered_set<int>(axes.begin(), axes.end());
  auto base_shape = this->_tensor->shape().get();
  std::vector<fl::Dim> new_shape;
  for (size_t idx = 0; idx < base_shape.size(); ++idx) {
    if (axes_set.count(idx) || (axes_set.size() == 0)) {
      if (keep_dims) {
        new_shape.emplace_back(1);
      }
      continue;
    }
    new_shape.emplace_back(base_shape[idx]);
  }
  const auto& shape = fl::Shape(new_shape);
  t = fl::reshape(t, shape);
  g_bytes_used += t.bytes();
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

Napi::Value Tensor::Var(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 3) {
    Napi::TypeError::New(info.Env(),
                         "Tensor method `var` expects exactly 2 args ")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsArray()) {
    Napi::TypeError::New(
        info.Env(),
        "Tensor method `var` expects 1st arg to be typeof `number[]`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[1].IsBoolean()) {
    Napi::TypeError::New(
        info.Env(), "Tensor method `var` expects 2nd arg to be typeof boolean")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[2].IsBoolean()) {
    Napi::TypeError::New(
        info.Env(), "Tensor method `var` expects 3rd arg to be typeof boolean")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Array axes_arg = info[0].As<Napi::Array>();
  bool bias = info[1].As<Napi::Boolean>().Value();
  bool keep_dims = info[2].As<Napi::Boolean>().Value();
  auto axes =
      jsArrayArg<int>(axes_arg, g_row_major, this->_tensor->ndim(), env);
  fl::Tensor t;
  t = fl::var(*(this->_tensor), axes, bias, keep_dims);
  auto axes_set = std::unordered_set<int>(axes.begin(), axes.end());
  auto base_shape = this->_tensor->shape().get();
  std::vector<fl::Dim> new_shape;
  for (size_t idx = 0; idx < base_shape.size(); ++idx) {
    if (axes_set.count(idx) || (axes_set.size() == 0)) {
      if (keep_dims) {
        new_shape.emplace_back(1);
      }
      continue;
    }
    new_shape.emplace_back(base_shape[idx]);
  }
  const auto& shape = fl::Shape(new_shape);
  t = fl::reshape(t, shape);
  g_bytes_used += t.bytes();
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

Napi::Value Tensor::Std(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2) {
    Napi::TypeError::New(info.Env(),
                         "Tensor method `std` expects exactly 2 args ")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsArray()) {
    Napi::TypeError::New(
        info.Env(),
        "Tensor method `std` expects 1st arg to be typeof `number[]`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[1].IsBoolean()) {
    Napi::TypeError::New(
        info.Env(), "Tensor method `std` expects 2nd arg to be typeof boolean")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Array axes_arg = info[0].As<Napi::Array>();
  bool keep_dims = info[1].As<Napi::Boolean>().Value();
  auto axes =
      jsArrayArg<int>(axes_arg, g_row_major, this->_tensor->ndim(), env);
  fl::Tensor t;
  t = fl::std(*(this->_tensor), axes, keep_dims);
  auto axes_set = std::unordered_set<int>(axes.begin(), axes.end());
  auto base_shape = this->_tensor->shape().get();
  std::vector<fl::Dim> new_shape;
  for (size_t idx = 0; idx < base_shape.size(); ++idx) {
    if (axes_set.count(idx) || (axes_set.size() == 0)) {
      if (keep_dims) {
        new_shape.emplace_back(1);
      }
      continue;
    }
    new_shape.emplace_back(base_shape[idx]);
  }
  const auto& shape = fl::Shape(new_shape);
  t = fl::reshape(t, shape);
  g_bytes_used += t.bytes();
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

Napi::Value Tensor::Norm(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 3) {
    Napi::TypeError::New(info.Env(),
                         "Tensor method `norm` expects exactly 2 args ")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsArray()) {
    Napi::TypeError::New(
        info.Env(),
        "Tensor method `norm` expects 1st arg to be typeof `number[]`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[1].IsNumber()) {
    Napi::TypeError::New(
        info.Env(),
        "Tensor method `norm` expects 2nd arg to be typeof `number`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[2].IsBoolean()) {
    Napi::TypeError::New(
        info.Env(),
        "Tensor method `std` expects 3rd arg to be typeof `boolean`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Array axes_arg = info[0].As<Napi::Array>();
  double p = info[1].As<Napi::Number>().DoubleValue();
  bool keep_dims = info[1].As<Napi::Boolean>().Value();
  auto axes =
      jsArrayArg<int>(axes_arg, g_row_major, this->_tensor->ndim(), env);
  fl::Tensor t;
  t = fl::norm(*(this->_tensor), axes, p, keep_dims);
  if (p == std::numeric_limits<double>::infinity()) {
    t = fl::abs(*(this->_tensor));
    t = fl::amax(t, axes, keep_dims);
  }
  auto axes_set = std::unordered_set<int>(axes.begin(), axes.end());
  auto base_shape = this->_tensor->shape().get();
  std::vector<fl::Dim> new_shape;
  for (size_t idx = 0; idx < base_shape.size(); ++idx) {
    if (axes_set.count(idx) || (axes_set.size() == 0)) {
      if (keep_dims) {
        new_shape.emplace_back(1);
      }
      continue;
    }
    new_shape.emplace_back(base_shape[idx]);
  }
  const auto& shape = fl::Shape(new_shape);
  t = fl::reshape(t, shape);
  g_bytes_used += t.bytes();
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

Napi::Value Tensor::CountNonZero(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2) {
    Napi::TypeError::New(info.Env(),
                         "Tensor method `countNonzero` expects exactly 2 args ")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsArray()) {
    Napi::TypeError::New(
        info.Env(),
        "Tensor method `countNonzero` expects 1st arg to be typeof `number[]`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[1].IsBoolean()) {
    Napi::TypeError::New(
        info.Env(),
        "Tensor method `countNonzero` expects 2nd arg to be typeof boolean")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Array axes_arg = info[0].As<Napi::Array>();
  bool keep_dims = info[1].As<Napi::Boolean>().Value();
  auto axes =
      jsArrayArg<int>(axes_arg, g_row_major, this->_tensor->ndim(), env);
  fl::Tensor t;
  t = fl::countNonzero(*(this->_tensor), axes, keep_dims);
  auto axes_set = std::unordered_set<int>(axes.begin(), axes.end());
  auto base_shape = this->_tensor->shape().get();
  std::vector<fl::Dim> new_shape;
  for (size_t idx = 0; idx < base_shape.size(); ++idx) {
    if (axes_set.count(idx) || (axes_set.size() == 0)) {
      if (keep_dims) {
        new_shape.emplace_back(1);
      }
      continue;
    }
    new_shape.emplace_back(base_shape[idx]);
  }
  const auto& shape = fl::Shape(new_shape);
  t = fl::reshape(t, shape);
  g_bytes_used += t.bytes();
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

Napi::Value Tensor::Any(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2) {
    Napi::TypeError::New(info.Env(),
                         "Tensor method `any` expects exactly 2 args ")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsArray()) {
    Napi::TypeError::New(
        info.Env(),
        "Tensor method `any` expects 1st arg to be typeof `number[]`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[1].IsBoolean()) {
    Napi::TypeError::New(
        info.Env(), "Tensor method `any` expects 2nd arg to be typeof boolean")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Array axes_arg = info[0].As<Napi::Array>();
  bool keep_dims = info[1].As<Napi::Boolean>().Value();
  auto axes =
      jsArrayArg<int>(axes_arg, g_row_major, this->_tensor->ndim(), env);
  fl::Tensor t;
  t = fl::any(*(this->_tensor), axes, keep_dims);
  auto axes_set = std::unordered_set<int>(axes.begin(), axes.end());
  auto base_shape = this->_tensor->shape().get();
  std::vector<fl::Dim> new_shape;
  for (size_t idx = 0; idx < base_shape.size(); ++idx) {
    if (axes_set.count(idx) || (axes_set.size() == 0)) {
      if (keep_dims) {
        new_shape.emplace_back(1);
      }
      continue;
    }
    new_shape.emplace_back(base_shape[idx]);
  }
  const auto& shape = fl::Shape(new_shape);
  t = fl::reshape(t, shape);
  g_bytes_used += t.bytes();
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

Napi::Value Tensor::All(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2) {
    Napi::TypeError::New(info.Env(),
                         "Tensor method `any` expects exactly 2 args ")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsArray()) {
    Napi::TypeError::New(
        info.Env(),
        "Tensor method `any` expects 1st arg to be typeof `number[]`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[1].IsBoolean()) {
    Napi::TypeError::New(
        info.Env(), "Tensor method `any` expects 2nd arg to be typeof boolean")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Array axes_arg = info[0].As<Napi::Array>();
  bool keep_dims = info[1].As<Napi::Boolean>().Value();
  auto axes =
      jsArrayArg<int>(axes_arg, g_row_major, this->_tensor->ndim(), env);
  fl::Tensor t;
  t = fl::all(*(this->_tensor), axes, keep_dims);
  auto axes_set = std::unordered_set<int>(axes.begin(), axes.end());
  auto base_shape = this->_tensor->shape().get();
  std::vector<fl::Dim> new_shape;
  for (size_t idx = 0; idx < base_shape.size(); ++idx) {
    if (axes_set.count(idx) || (axes_set.size() == 0)) {
      if (keep_dims) {
        new_shape.emplace_back(1);
      }
      continue;
    }
    new_shape.emplace_back(base_shape[idx]);
  }
  const auto& shape = fl::Shape(new_shape);
  t = fl::reshape(t, shape);
  g_bytes_used += t.bytes();
  auto* tensor = new fl::Tensor(t);
  auto wrapped = Napi::External<fl::Tensor>::New(env, tensor);
  Napi::Value wrappedTensor = Tensor::constructor->New({wrapped});
  return wrappedTensor;
}

// define the `Tensor` class NAPI export
Napi::FunctionReference* Tensor::constructor;
Napi::Function Tensor::GetClass(Napi::Env env) {
  Napi::Function func = DefineClass(
      env, "Tensor",
      {
          Tensor::InstanceMethod("elements", &Tensor::Elements),
          Tensor::InstanceMethod("ndim", &Tensor::NDim),
          Tensor::InstanceMethod("dtype", &Tensor::Dtype),
          Tensor::InstanceMethod("bytes", &Tensor::Bytes),
          Tensor::InstanceMethod("shape", &Tensor::Shape),
          Tensor::InstanceMethod("shape64", &Tensor::Shape64),
          Tensor::InstanceMethod("toString", &Tensor::ToString),
          Tensor::InstanceMethod("update", &Tensor::Update),
          Tensor::InstanceMethod("eval", &Tensor::Eval),
          Tensor::InstanceMethod("save", &Tensor::Save),
          Tensor::InstanceMethod("toFloat32Array", &Tensor::ToFloat32Array),
          Tensor::InstanceMethod("toFloat64Array", &Tensor::ToFloat64Array),
          Tensor::InstanceMethod("toBoolInt8Array", &Tensor::ToBoolInt8Array),
          Tensor::InstanceMethod("toInt16Array", &Tensor::ToInt16Array),
          Tensor::InstanceMethod("toInt32Array", &Tensor::ToInt32Array),
          Tensor::InstanceMethod("toBigInt64Array", &Tensor::ToBigInt64Array),
          Tensor::InstanceMethod("toUint8Array", &Tensor::ToUint8Array),
          Tensor::InstanceMethod("toUint16Array", &Tensor::ToUint16Array),
          Tensor::InstanceMethod("toUint32Array", &Tensor::ToUint32Array),
          Tensor::InstanceMethod("toBigUint64Array", &Tensor::ToBigUint64Array),
          Tensor::InstanceMethod("dispose", &Tensor::Dispose),
          Tensor::InstanceMethod("reshape", &Tensor::Reshape),
          Tensor::InstanceMethod("astype", &Tensor::AsType),
          Tensor::InstanceMethod("transpose", &Tensor::Transpose),
          Tensor::InstanceMethod("tile", &Tensor::Tile),
          Tensor::InstanceMethod("nonzero", &Tensor::NonZero),
          Tensor::InstanceMethod("negative", &Tensor::Negative),
          Tensor::InstanceMethod("logicalNot", &Tensor::LogicalNot),
          Tensor::InstanceMethod("exp", &Tensor::Exp),
          Tensor::InstanceMethod("log", &Tensor::Log),
          Tensor::InstanceMethod("log1p", &Tensor::Log1p),
          Tensor::InstanceMethod("sin", &Tensor::Sin),
          Tensor::InstanceMethod("cos", &Tensor::Cos),
          Tensor::InstanceMethod("sqrt", &Tensor::Sqrt),
          Tensor::InstanceMethod("tanh", &Tensor::Tanh),
          Tensor::InstanceMethod("floor", &Tensor::Floor),
          Tensor::InstanceMethod("ceil", &Tensor::Ceil),
          Tensor::InstanceMethod("rint", &Tensor::Rint),
          Tensor::InstanceMethod("absolute", &Tensor::Absolute),
          Tensor::InstanceMethod("sigmoid", &Tensor::Sigmoid),
          Tensor::InstanceMethod("erf", &Tensor::Erf),
          Tensor::InstanceMethod("flip", &Tensor::Flip),
          Tensor::InstanceMethod("clip", &Tensor::Clip),
          Tensor::InstanceMethod("roll", &Tensor::Roll),
          Tensor::InstanceMethod("isnan", &Tensor::IsNaN),
          Tensor::InstanceMethod("isinf", &Tensor::IsInf),
          Tensor::InstanceMethod("sign", &Tensor::Sign),
          Tensor::InstanceMethod("tril", &Tensor::Tril),
          Tensor::InstanceMethod("triu", &Tensor::Triu),
          Tensor::InstanceMethod("where", &Tensor::Where),
          Tensor::InstanceMethod("sort", &Tensor::Sort),
          Tensor::InstanceMethod("add", &Tensor::Add),
          Tensor::InstanceMethod("sub", &Tensor::Sub),
          Tensor::InstanceMethod("mul", &Tensor::Mul),
          Tensor::InstanceMethod("div", &Tensor::Div),
          Tensor::InstanceMethod("eq", &Tensor::Eq),
          Tensor::InstanceMethod("neq", &Tensor::Neq),
          Tensor::InstanceMethod("lessThan", &Tensor::LessThan),
          Tensor::InstanceMethod("lessThanEqual", &Tensor::LessThanEqual),
          Tensor::InstanceMethod("greaterThan", &Tensor::GreaterThan),
          Tensor::InstanceMethod("greaterThanEqual", &Tensor::GreaterThanEqual),
          Tensor::InstanceMethod("logicalOr", &Tensor::LogicalOr),
          Tensor::InstanceMethod("logicalAnd", &Tensor::LogicalAnd),
          Tensor::InstanceMethod("mod", &Tensor::Mod),
          Tensor::InstanceMethod("bitwiseAnd", &Tensor::BitwiseAnd),
          Tensor::InstanceMethod("bitwiseOr", &Tensor::BitwiseOr),
          Tensor::InstanceMethod("bitwiseXor", &Tensor::BitwiseXor),
          Tensor::InstanceMethod("lShift", &Tensor::LShift),
          Tensor::InstanceMethod("rShift", &Tensor::RShift),
          Tensor::InstanceMethod("minimum", &Tensor::Minimum),
          Tensor::InstanceMethod("maximum", &Tensor::Maximum),
          Tensor::InstanceMethod("power", &Tensor::Power),
          Tensor::InstanceMethod("matmul", &Tensor::MatMul),
          Tensor::InstanceMethod("conv2d", &Tensor::Conv2d),
          Tensor::InstanceMethod("amin", &Tensor::AMin),
          Tensor::InstanceMethod("amax", &Tensor::AMax),
          Tensor::InstanceMethod("argmin", &Tensor::ArgMin),
          Tensor::InstanceMethod("argmax", &Tensor::ArgMax),
          Tensor::InstanceMethod("sum", &Tensor::Sum),
          Tensor::InstanceMethod("cumsum", &Tensor::CumSum),
          Tensor::InstanceMethod("mean", &Tensor::Mean),
          Tensor::InstanceMethod("median", &Tensor::Median),
          Tensor::InstanceMethod("_var", &Tensor::Var),
          Tensor::InstanceMethod("std", &Tensor::Std),
          Tensor::InstanceMethod("norm", &Tensor::Norm),
          Tensor::InstanceMethod("countNonzero", &Tensor::CountNonZero),
          Tensor::InstanceMethod("any", &Tensor::Any),
          Tensor::InstanceMethod("all", &Tensor::All),
      });

  constructor = new Napi::FunctionReference();
  *constructor = Napi::Persistent(func);  // <-- and here
  return func;
}

// define NAPI exports (class, functions, etc)
Napi::Object Init(Napi::Env env, Napi::Object exports) {
  exports.Set(Napi::String::New(env, "Tensor"), Tensor::GetClass(env));
  exports.Set(Napi::String::New(env, "setRowMajor"),
              Napi::Function::New(env, setRowMajor));
  exports.Set(Napi::String::New(env, "setColMajor"),
              Napi::Function::New(env, setColMajor));
  exports.Set(Napi::String::New(env, "init"), Napi::Function::New(env, init));
  exports.Set(Napi::String::New(env, "bytesUsed"),
              Napi::Function::New(env, bytesUsed));
  exports.Set(Napi::String::New(env, "isRowMajor"),
              Napi::Function::New(env, isRowMajor));
  exports.Set(Napi::String::New(env, "isColMajor"),
              Napi::Function::New(env, isColMajor));
  exports.Set(Napi::String::New(env, "dtypeFloat32"),
              Napi::Function::New(env, DtypeFloat32));
  exports.Set(Napi::String::New(env, "dtypeFloat64"),
              Napi::Function::New(env, DtypeFloat64));
  exports.Set(Napi::String::New(env, "dtypeBoolInt8"),
              Napi::Function::New(env, DtypeBoolInt8));
  exports.Set(Napi::String::New(env, "dtypeInt16"),
              Napi::Function::New(env, DtypeInt16));
  exports.Set(Napi::String::New(env, "dtypeInt32"),
              Napi::Function::New(env, DtypeInt32));
  exports.Set(Napi::String::New(env, "dtypeInt64"),
              Napi::Function::New(env, DtypeInt64));
  exports.Set(Napi::String::New(env, "dtypeUint8"),
              Napi::Function::New(env, DtypeUint8));
  exports.Set(Napi::String::New(env, "dtypeUint16"),
              Napi::Function::New(env, DtypeUint16));
  exports.Set(Napi::String::New(env, "dtypeUint32"),
              Napi::Function::New(env, DtypeUint32));
  exports.Set(Napi::String::New(env, "dtypeUint64"),
              Napi::Function::New(env, DtypeUint64));
  exports.Set(Napi::String::New(env, "_rand"), Napi::Function::New(env, Rand));
  exports.Set(Napi::String::New(env, "_randn"),
              Napi::Function::New(env, Randn));
  exports.Set(Napi::String::New(env, "_full"), Napi::Function::New(env, Full));
  exports.Set(Napi::String::New(env, "_identity"),
              Napi::Function::New(env, Identity));
  exports.Set(Napi::String::New(env, "_arange"),
              Napi::Function::New(env, Arange));
  exports.Set(Napi::String::New(env, "_iota"), Napi::Function::New(env, Iota));
  exports.Set(Napi::String::New(env, "_reshape"),
              Napi::Function::New(env, Reshape));
  exports.Set(Napi::String::New(env, "_transpose"),
              Napi::Function::New(env, Transpose));
  exports.Set(Napi::String::New(env, "_tile"), Napi::Function::New(env, Tile));
  exports.Set(Napi::String::New(env, "_concatenate"),
              Napi::Function::New(env, Concatenate));
  exports.Set(Napi::String::New(env, "_nonzero"),
              Napi::Function::New(env, NonZero));
  exports.Set(Napi::String::New(env, "_negative"),
              Napi::Function::New(env, Negative));
  exports.Set(Napi::String::New(env, "_logicalNot"),
              Napi::Function::New(env, LogicalNot));
  exports.Set(Napi::String::New(env, "_exp"), Napi::Function::New(env, Exp));
  exports.Set(Napi::String::New(env, "_log"), Napi::Function::New(env, Log));
  exports.Set(Napi::String::New(env, "_log1p"),
              Napi::Function::New(env, Log1p));
  exports.Set(Napi::String::New(env, "_sin"), Napi::Function::New(env, Sin));
  exports.Set(Napi::String::New(env, "_cos"), Napi::Function::New(env, Cos));
  exports.Set(Napi::String::New(env, "_sqrt"), Napi::Function::New(env, Sqrt));
  exports.Set(Napi::String::New(env, "_tanh"), Napi::Function::New(env, Tanh));
  exports.Set(Napi::String::New(env, "_floor"),
              Napi::Function::New(env, Floor));
  exports.Set(Napi::String::New(env, "_ceil"), Napi::Function::New(env, Ceil));
  exports.Set(Napi::String::New(env, "_rint"), Napi::Function::New(env, Rint));
  exports.Set(Napi::String::New(env, "_absolute"),
              Napi::Function::New(env, Absolute));
  exports.Set(Napi::String::New(env, "_sigmoid"),
              Napi::Function::New(env, Sigmoid));
  exports.Set(Napi::String::New(env, "_erf"), Napi::Function::New(env, Erf));
  exports.Set(Napi::String::New(env, "_flip"), Napi::Function::New(env, Flip));
  exports.Set(Napi::String::New(env, "_clip"), Napi::Function::New(env, Clip));
  exports.Set(Napi::String::New(env, "_roll"), Napi::Function::New(env, Roll));
  exports.Set(Napi::String::New(env, "_isnan"),
              Napi::Function::New(env, IsNaN));
  exports.Set(Napi::String::New(env, "_isinf"),
              Napi::Function::New(env, IsInf));
  exports.Set(Napi::String::New(env, "_sign"), Napi::Function::New(env, Sign));
  exports.Set(Napi::String::New(env, "_tril"), Napi::Function::New(env, Tril));
  exports.Set(Napi::String::New(env, "_triu"), Napi::Function::New(env, Triu));
  exports.Set(Napi::String::New(env, "_where"),
              Napi::Function::New(env, Where));
  exports.Set(Napi::String::New(env, "_sort"), Napi::Function::New(env, Sort));
  exports.Set(Napi::String::New(env, "_add"), Napi::Function::New(env, Add));
  exports.Set(Napi::String::New(env, "_sub"), Napi::Function::New(env, Sub));
  exports.Set(Napi::String::New(env, "_mul"), Napi::Function::New(env, Mul));
  exports.Set(Napi::String::New(env, "_div"), Napi::Function::New(env, Div));
  exports.Set(Napi::String::New(env, "_eq"), Napi::Function::New(env, Eq));
  exports.Set(Napi::String::New(env, "_neq"), Napi::Function::New(env, Neq));
  exports.Set(Napi::String::New(env, "_lessThan"),
              Napi::Function::New(env, LessThan));
  exports.Set(Napi::String::New(env, "_lessThanEqual"),
              Napi::Function::New(env, LessThanEqual));
  exports.Set(Napi::String::New(env, "_greaterThan"),
              Napi::Function::New(env, GreaterThan));
  exports.Set(Napi::String::New(env, "_greaterThanEqual"),
              Napi::Function::New(env, GreaterThanEqual));
  exports.Set(Napi::String::New(env, "_logicalOr"),
              Napi::Function::New(env, LogicalOr));
  exports.Set(Napi::String::New(env, "_logicalAnd"),
              Napi::Function::New(env, LogicalAnd));
  exports.Set(Napi::String::New(env, "_mod"), Napi::Function::New(env, Mod));
  exports.Set(Napi::String::New(env, "_bitwiseAnd"),
              Napi::Function::New(env, BitwiseAnd));
  exports.Set(Napi::String::New(env, "_bitwiseOr"),
              Napi::Function::New(env, BitwiseOr));
  exports.Set(Napi::String::New(env, "_bitwiseXor"),
              Napi::Function::New(env, BitwiseXor));
  exports.Set(Napi::String::New(env, "_lShift"),
              Napi::Function::New(env, LShift));
  exports.Set(Napi::String::New(env, "_rShift"),
              Napi::Function::New(env, RShift));
  exports.Set(Napi::String::New(env, "_minimum"),
              Napi::Function::New(env, Minimum));
  exports.Set(Napi::String::New(env, "_maximum"),
              Napi::Function::New(env, Maximum));
  exports.Set(Napi::String::New(env, "_power"),
              Napi::Function::New(env, Power));
  exports.Set(Napi::String::New(env, "_matmul"),
              Napi::Function::New(env, MatMul));
  exports.Set(Napi::String::New(env, "_conv2d"),
              Napi::Function::New(env, Conv2d));
  exports.Set(Napi::String::New(env, "_amin"), Napi::Function::New(env, AMin));
  exports.Set(Napi::String::New(env, "_amax"), Napi::Function::New(env, AMax));
  exports.Set(Napi::String::New(env, "_argmin"),
              Napi::Function::New(env, ArgMin));
  exports.Set(Napi::String::New(env, "_argmax"),
              Napi::Function::New(env, ArgMax));
  exports.Set(Napi::String::New(env, "_sum"), Napi::Function::New(env, Sum));
  exports.Set(Napi::String::New(env, "_cumsum"),
              Napi::Function::New(env, CumSum));
  exports.Set(Napi::String::New(env, "_mean"), Napi::Function::New(env, Mean));
  exports.Set(Napi::String::New(env, "_median"),
              Napi::Function::New(env, Median));
  exports.Set(Napi::String::New(env, "_variance"),
              Napi::Function::New(env, Var));
  exports.Set(Napi::String::New(env, "_std"), Napi::Function::New(env, Std));
  exports.Set(Napi::String::New(env, "_norm"), Napi::Function::New(env, Norm));
  exports.Set(Napi::String::New(env, "_countNonzero"),
              Napi::Function::New(env, CountNonZero));
  exports.Set(Napi::String::New(env, "_any"), Napi::Function::New(env, Any));
  exports.Set(Napi::String::New(env, "_all"), Napi::Function::New(env, All));
  return exports;
}

NODE_API_MODULE(addon, Init)
