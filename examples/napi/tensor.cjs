const {
  Tensor: _Tensor,
  dtypeFloat32,
  dtypeFloat64,
  dtypeBoolInt8,
  dtypeInt16,
  dtypeInt32,
  dtypeInt64,
  dtypeUint8,
  dtypeUint16,
  dtypeUint32,
  dtypeUint64
} = require('../../build/Release/flashlight_napi_bindings')

const dtype = {
  Float32: dtypeFloat32(),
  Float64: dtypeFloat64(),
  BoolInt8: dtypeBoolInt8(),
  Int8: dtypeBoolInt8(),
  Int16: dtypeInt16(),
  Int32: dtypeInt32(),
  Int64: dtypeInt64(),
  BigInt64: dtypeInt64(),
  Uint8: dtypeUint8(),
  Uint16: dtypeUint16(),
  Uint32: dtypeUint32(),
  Uint64: dtypeUint64(),
  BigUint64: dtypeUint64()
}

class Tensor {
  constructor(obj) {
    this._napi_tensor = new _Tensor(obj)
  }

  get elements() {
    return this._napi_tensor.elements()
  }

  get ndim() {
    return this._napi_tensor.ndim()
  }

  get dtype() {
    return this._napi_tensor.dtype()
  }

  get bytes() {
    return this._napi_tensor.bytes()
  }

  get shape() {
    return this._napi_tensor.shape()
  }

  get shape64() {
    return this._napi_tensor.shape64()
  }

  toString() {
    return this._napi_tensor.toString()
  }

  update(tensor) {
    this._deps = tensor._deps
    this._napi_tensor.update(tensor._napi_tensor._tensor)
    if (this._checkpoint_file) {
      if (this._checkpoint_callback()) {
        this._napi_tensor.save(this._checkpoint_file)
      }
    }
  }

  checkpoint(file, callback) {
    if (file instanceof Function) {
      callback = file
      file = undefined
    }
    if (file === undefined) {
      this._checkpoint_file = `tensor_${cyrb53(getStack(true))}.fl`
    } else {
      this._checkpoint_file = file.toString()
    }
    if (callback !== undefined) {
      this._checkpoint_callback = callback
    } else {
      this._checkpoint_callback = () => true
    }
    if (existsSync(this._checkpoint_file)) {
      this.update(new Tensor(this._checkpoint_file))
    } else {
      this.save(this._checkpoint_file)
    }
    return this
  }

  dispose() {
    this._napi_tensor.dispose()
  }

  eval() {
    this._napi_tensor.eval()
  }

  save(file) {
    this._napi_tensor.save(file)
  }

  toFloat32Array() {
    return this._napi_tensor.toFloat32Array()
  }

  toFloat64Array() {
    return this._napi_tensor.toFloat64Array()
  }

  toBoolInt8Array() {
    return this._napi_tensor.toBoolInt8Array()
  }

  toInt16Array() {
    return this._napi_tensor.toInt16Array()
  }

  toInt32Array() {
    return this._napi_tensor.toInt32Array()
  }

  toBigInt64Array() {
    return this._napi_tensor.toBigInt64Array()
  }

  toUint8Array() {
    return this._napi_tensor.toUint8Array()
  }

  toUint16Array() {
    return this._napi_tensor.toUint16Array()
  }

  toUint32Array() {
    return this._napi_tensor.toUint32Array()
  }

  toBigUint64Array() {
    return this._napi_tensor.toBigUint64Array()
  }

  reshape(shape) {
    return new Tensor(this._napi_tensor.reshape(shape))
  }
}

module.exports = { Tensor, dtype }
