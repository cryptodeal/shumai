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

std::vector<long long> jsArrayArg(Napi::Array arr,
                                  bool reverse,
                                  int invert,
                                  Napi::Env env) {
  std::vector<long long> out;
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
      std::cout << v << std::endl;
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
      jsArrayArg(info[0].As<Napi::Array>(), g_row_major, false, env);
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
      jsArrayArg(axes_arg, g_row_major, this->_tensor->ndim(), env);

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
  std::vector<long long> shape = jsArrayArg(shape_arg, g_row_major, false, env);
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
  return exports;
}

NODE_API_MODULE(addon, Init)
