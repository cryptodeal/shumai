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

  astype(dtype) {
    return new Tensor(this._napi_tensor.astype(dtype))
  }

  transpose(axes) {
    return new Tensor(this._napi_tensor.transpose(axes))
  }

  tile(shape) {
    return new Tensor(this._napi_tensor.tile(shape))
  }

  nonzero() {
    return new Tensor(this._napi_tensor.nonzero())
  }

  negative() {
    return new Tensor(this._napi_tensor.negative())
  }

  negate() {
    return new Tensor(this._napi_tensor.negative())
  }

  logicalNot() {
    return new Tensor(this._napi_tensor.logicalNot())
  }

  exp() {
    return new Tensor(this._napi_tensor.exp())
  }

  log() {
    return new Tensor(this._napi_tensor.log())
  }

  sin() {
    return new Tensor(this._napi_tensor.sin())
  }

  cos() {
    return new Tensor(this._napi_tensor.cos())
  }

  sqrt() {
    return new Tensor(this._napi_tensor.sqrt())
  }

  floor() {
    return new Tensor(this._napi_tensor.floor())
  }

  ceil() {
    return new Tensor(this._napi_tensor.ceil())
  }

  rint() {
    return new Tensor(this._napi_tensor.rint())
  }

  absolute() {
    return new Tensor(this._napi_tensor.absolute())
  }

  abs() {
    return new Tensor(this._napi_tensor.absolute())
  }

  sigmoid() {
    return new Tensor(this._napi_tensor.sigmoid())
  }

  erf() {
    return new Tensor(this._napi_tensor.erf())
  }

  flip(dim) {
    return new Tensor(this._napi_tensor.flip(dim))
  }

  clip(low, high) {
    return new Tensor(this._napi_tensor.clip(low._napi_tensor, high._napi_tensor))
  }

  roll(shift, axis) {
    return new Tensor(this._napi_tensor.roll(shift, axis))
  }

  isnan() {
    return new Tensor(this._napi_tensor.isnan())
  }

  isinf() {
    return new Tensor(this._napi_tensor.isinf())
  }

  sign() {
    return new Tensor(this._napi_tensor.sign())
  }

  tril() {
    return new Tensor(this._napi_tensor.tril())
  }

  triu() {
    return new Tensor(this._napi_tensor.triu())
  }

  where(x, y) {
    return new Tensor(this._napi_tensor.where(x._napi_tensor, y._napi_tensor))
  }

  sort(dim) {
    return new Tensor(this._napi_tensor.sort(dim))
  }

  add(other) {
    return new Tensor(this._napi_tensor.add(other._napi_tensor))
  }

  sub(other) {
    return new Tensor(this._napi_tensor.add(other._napi_tensor))
  }

  mul(other) {
    return new Tensor(this._napi_tensor.mul(other._napi_tensor))
  }

  div(other) {
    return new Tensor(this._napi_tensor.div(other._napi_tensor))
  }

  eq(other) {
    return new Tensor(this._napi_tensor.eq(other._napi_tensor))
  }

  neq(other) {
    return new Tensor(this._napi_tensor.neq(other._napi_tensor))
  }

  lessThan(other) {
    return new Tensor(this._napi_tensor.lessThan(other._napi_tensor))
  }

  lt(other) {
    return new Tensor(this._napi_tensor.lessThan(other._napi_tensor))
  }

  lessThanEqual(other) {
    return new Tensor(this._napi_tensor.lessThanEqual(other._napi_tensor))
  }

  lte(other) {
    return new Tensor(this._napi_tensor.lessThanEqual(other._napi_tensor))
  }

  greaterThan(other) {
    return new Tensor(this._napi_tensor.greaterThan(other._napi_tensor))
  }

  gt(other) {
    return new Tensor(this._napi_tensor.greaterThan(other._napi_tensor))
  }

  greaterThanEqual(other) {
    return new Tensor(this._napi_tensor.greaterThanEqual(other._napi_tensor))
  }

  gte(other) {
    return new Tensor(this._napi_tensor.greaterThanEqual(other._napi_tensor))
  }

  logicalOr(other) {
    return new Tensor(this._napi_tensor.logicalOr(other._napi_tensor))
  }

  logicalAnd(other) {
    return new Tensor(this._napi_tensor.logicalAnd(other._napi_tensor))
  }

  mod(other) {
    return new Tensor(this._napi_tensor.mod(other._napi_tensor))
  }

  bitwiseAnd(other) {
    return new Tensor(this._napi_tensor.bitwiseAnd(other._napi_tensor))
  }

  bitwiseOr(other) {
    return new Tensor(this._napi_tensor.bitwiseOr(other._napi_tensor))
  }

  bitwiseXor(other) {
    return new Tensor(this._napi_tensor.bitwiseXor(other._napi_tensor))
  }

  lShift(other) {
    return new Tensor(this._napi_tensor.lShift(other._napi_tensor))
  }

  rShift(other) {
    return new Tensor(this._napi_tensor.rShift(other._napi_tensor))
  }

  minimum(other) {
    return new Tensor(this._napi_tensor.minimum(other._napi_tensor))
  }

  maximum(other) {
    return new Tensor(this._napi_tensor.maximum(other._napi_tensor))
  }

  power(other) {
    return new Tensor(this._napi_tensor.power(other._napi_tensor))
  }

  pow(other) {
    return new Tensor(this._napi_tensor.power(other._napi_tensor))
  }

  matmul(other) {
    return new Tensor(this._napi_tensor.matmul(other._napi_tensor))
  }

  mm(other) {
    return new Tensor(this._napi_tensor.matmul(other._napi_tensor))
  }

  conv2d(weights, sx, sy, px, py, dx, dy, groups) {
    return new Tensor(
      this._napi_tensor.conv2d(weights._napi_tensor, sx, sy, px, py, dx, dy, groups)
    )
  }

  amin(axes, keepdims) {
    return new Tensor(this._napi_tensor.amin(axes, keepdims))
  }

  amax(axes, keepdims) {
    return new Tensor(this._napi_tensor.amax(axes, keepdims))
  }

  argmin(axis, keepdims) {
    return new Tensor(this._napi_tensor.argmin(axis, keepdims))
  }

  argmax(axis, keepdims) {
    return new Tensor(this._napi_tensor.argmax(axis, keepdims))
  }

  sum(axes, keepdims) {
    return new Tensor(this._napi_tensor.sum(axes, keepdims))
  }

  cumsum(axis) {
    return new Tensor(this._napi_tensor.cumsum(axis))
  }

  mean(axes, keepdims) {
    return new Tensor(this._napi_tensor.mean(axes, keepdims))
  }

  median(axes, keepdims) {
    return new Tensor(this._napi_tensor.mean(axes, keepdims))
  }

  _var(axes, bias, keepdims) {
    return new Tensor(this._napi_tensor._var(axes, bias, keepdims))
  }

  variance(axes, bias, keepdims) {
    return new Tensor(this._napi_tensor._var(axes, bias, keepdims))
  }

  std(axes, keepdims) {
    return new Tensor(this._napi_tensor.std(axes, keepdims))
  }

  norm(axes, p, keepdims) {
    return new Tensor(this._napi_tensor.norm(axes, p, keepdims))
  }

  countNonzero(axes, keepdims) {
    return new Tensor(this._napi_tensor.countNonzero(axes, keepdims))
  }

  any(axes, keepdims) {
    return new Tensor(this._napi_tensor.any(axes, keepdims))
  }

  all(axes, keepdims) {
    return new Tensor(this._napi_tensor.all(axes, keepdims))
  }
}

module.exports = { Tensor, dtype }
