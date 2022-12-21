import { median } from '@shumai/shumai'
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
  dtypeUint64,
  _rand,
  _randn,
  _full,
  _identity,
  _arange,
  _iota,
  _reshape,
  _transpose,
  _tile,
  _concatenate,
  _nonzero,
  _negative,
  _logicalNot,
  _exp,
  _log,
  _log1p,
  _sin,
  _cos,
  _sqrt,
  _tanh,
  _floor,
  _ceil,
  _rint,
  _absolute,
  _sigmoid,
  _erf,
  _flip,
  _clip,
  _roll,
  _isnan,
  _isinf,
  _sign,
  _tril,
  _triu,
  _where,
  _sort,
  _add,
  _sub,
  _mul,
  _div,
  _eq,
  _neq,
  _lessThan,
  _lessThanEqual,
  _greaterThan,
  _greaterThanEqual,
  _logicalOr,
  _logicalAnd,
  _mod,
  _bitwiseAnd,
  _bitwiseOr,
  _bitwiseXor,
  _lShift,
  _rShift,
  _minimum,
  _maximum,
  _power,
  _matmul,
  _conv2d,
  _amin,
  _amax,
  _argmin,
  _argmax,
  _sum,
  _cumsum,
  _mean,
  _median,
  _var,
  _std,
  _norm,
  _countNonzero,
  _any,
  _all
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

const rand = (shape) => {
  return new Tensor(_rand(shape))
}

const randn = (shape) => {
  return new Tensor(_randn(shape))
}

const full = (shape, val) => {
  return new Tensor(_full(shape, val))
}

const identity = (dim) => {
  return new Tensor(_identity(dim))
}

const ident = (dim) => {
  return new Tensor(_identity(dim))
}

const eye = (dim) => {
  return new Tensor(_identity(dim))
}

const arange = (start, end, step) => {
  return new Tensor(_arange(start, end, step))
}

const iota = (dims, tileDims) => {
  return new Tensor(_iota(dims, tileDims))
}

const reshape = (tensor, shape) => {
  return new Tensor(_reshape(tensor._napi_tensor, shape))
}

const transpose = (tensor, axes) => {
  return new Tensor(_transpose(tensor._napi_tensor, axes))
}

const tile = (tensor, shape) => {
  return new Tensor(_tile(tensor._napi_tensor, shape))
}

const concatenate = (tensors, axis) => {
  return new Tensor(
    _concatenate(
      tensors.map((t) => t._napi_tensor),
      axis
    )
  )
}

const concat = (tensors, axis) => {
  return new Tensor(
    _concatenate(
      tensors.map((t) => t._napi_tensor),
      axis
    )
  )
}

const nonzero = (tensor) => {
  return new Tensor(_nonzero(tensor._napi_tensor))
}

const negative = (tensor) => {
  return new Tensor(_negative(tensor._napi_tensor))
}

const negate = (tensor) => {
  return new Tensor(_negative(tensor._napi_tensor))
}

const logicalNot = (tensor) => {
  return new Tensor(_logicalNot(tensor._napi_tensor))
}

const exp = (tensor) => {
  return new Tensor(_exp(tensor._napi_tensor))
}

const log = (tensor) => {
  return new Tensor(_log(tensor._napi_tensor))
}

const log1p = (tensor) => {
  return new Tensor(_log1p(tensor._napi_tensor))
}

const sin = (tensor) => {
  return new Tensor(_sin(tensor._napi_tensor))
}

const cos = (tensor) => {
  return new Tensor(_cos(tensor._napi_tensor))
}

const sqrt = (tensor) => {
  return new Tensor(_sqrt(tensor._napi_tensor))
}

const tanh = (tensor) => {
  return new Tensor(_tanh(tensor._napi_tensor))
}

const floor = (tensor) => {
  return new Tensor(_floor(tensor._napi_tensor))
}

const ceil = (tensor) => {
  return new Tensor(_ceil(tensor._napi_tensor))
}

const rint = (tensor) => {
  return new Tensor(_rint(tensor._napi_tensor))
}

const absolute = (tensor) => {
  return new Tensor(_absolute(tensor._napi_tensor))
}

const abs = (tensor) => {
  return new Tensor(_absolute(tensor._napi_tensor))
}

const sigmoid = (tensor) => {
  return new Tensor(_sigmoid(tensor._napi_tensor))
}

const erf = (tensor) => {
  return new Tensor(_erf(tensor._napi_tensor))
}

const flip = (tensor, dim) => {
  return new Tensor(_flip(tensor._napi_tensor, dim))
}

const clip = (tensor, dim) => {
  return new Tensor(_clip(tensor._napi_tensor, dim))
}

const roll = (tensor, shift, axis) => {
  return new Tensor(_roll(tensor._napi_tensor, shift, axis))
}

const isnan = (tensor) => {
  return new Tensor(_isnan(tensor._napi_tensor))
}

const isinf = (tensor) => {
  return new Tensor(_isinf(tensor._napi_tensor))
}

const sign = (tensor) => {
  return new Tensor(_sign(tensor._napi_tensor))
}

const tril = (tensor) => {
  return new Tensor(_tril(tensor._napi_tensor))
}

const triu = (tensor) => {
  return new Tensor(_triu(tensor._napi_tensor))
}

const where = (cond, x, y) => {
  return new Tensor(_where(cond._napi_tensor, x._napi_tensor, y._napi_tensor))
}

const sort = (tensor, dim) => {
  return new Tensor(_sort(tensor._napi_tensor, dim))
}

const add = (tensor, other) => {
  return new Tensor(_add(tensor._napi_tensor, other._napi_tensor))
}

const sub = (tensor, other) => {
  return new Tensor(_sub(tensor._napi_tensor, other._napi_tensor))
}

const mul = (tensor, other) => {
  return new Tensor(_mul(tensor._napi_tensor, other._napi_tensor))
}

const div = (tensor, other) => {
  return new Tensor(_div(tensor._napi_tensor, other._napi_tensor))
}

const eq = (tensor, other) => {
  return new Tensor(_eq(tensor._napi_tensor, other._napi_tensor))
}

const neq = (tensor, other) => {
  return new Tensor(_neq(tensor._napi_tensor, other._napi_tensor))
}

const lessThan = (tensor, other) => {
  return new Tensor(_lessThan(tensor._napi_tensor, other._napi_tensor))
}

const lt = (tensor, other) => {
  return new Tensor(_lessThan(tensor._napi_tensor, other._napi_tensor))
}

const lessThanEqual = (tensor, other) => {
  return new Tensor(_lessThanEqual(tensor._napi_tensor, other._napi_tensor))
}

const lte = (tensor, other) => {
  return new Tensor(_lessThanEqual(tensor._napi_tensor, other._napi_tensor))
}

const greaterThan = (tensor, other) => {
  return new Tensor(_greaterThan(tensor._napi_tensor, other._napi_tensor))
}

const gt = (tensor, other) => {
  return new Tensor(_greaterThan(tensor._napi_tensor, other._napi_tensor))
}

const greaterThanEqual = (tensor, other) => {
  return new Tensor(_greaterThanEqual(tensor._napi_tensor, other._napi_tensor))
}

const gte = (tensor, other) => {
  return new Tensor(_greaterThanEqual(tensor._napi_tensor, other._napi_tensor))
}

const logicalOr = (tensor, other) => {
  return new Tensor(_logicalOr(tensor._napi_tensor, other._napi_tensor))
}

const logicalAnd = (tensor, other) => {
  return new Tensor(_logicalAnd(tensor._napi_tensor, other._napi_tensor))
}

const mod = (tensor, other) => {
  return new Tensor(_mod(tensor._napi_tensor, other._napi_tensor))
}

const bitwiseAnd = (tensor, other) => {
  return new Tensor(_bitwiseAnd(tensor._napi_tensor, other._napi_tensor))
}

const bitwiseOr = (tensor, other) => {
  return new Tensor(_bitwiseOr(tensor._napi_tensor, other._napi_tensor))
}

const bitwiseXor = (tensor, other) => {
  return new Tensor(_bitwiseXor(tensor._napi_tensor, other._napi_tensor))
}

const lShift = (tensor, other) => {
  return new Tensor(_lShift(tensor._napi_tensor, other._napi_tensor))
}

const rShift = (tensor, other) => {
  return new Tensor(_rShift(tensor._napi_tensor, other._napi_tensor))
}

const minimum = (tensor, other) => {
  return new Tensor(_minimum(tensor._napi_tensor, other._napi_tensor))
}

const maximum = (tensor, other) => {
  return new Tensor(_maximum(tensor._napi_tensor, other._napi_tensor))
}

const power = (tensor, other) => {
  return new Tensor(_power(tensor._napi_tensor, other._napi_tensor))
}

const pow = (tensor, other) => {
  return new Tensor(_power(tensor._napi_tensor, other._napi_tensor))
}

const matmul = (tensor, other) => {
  return new Tensor(_matmul(tensor._napi_tensor, other._napi_tensor))
}

const mm = (tensor, other) => {
  return new Tensor(_matmul(tensor._napi_tensor, other._napi_tensor))
}

const conv2d = (tensor, weights, sx, sy, px, py, dx, dy, groups) => {
  return new Tensor(
    _conv2d(tensor._napi_tensor, weights._napi_tensor, sx, sy, px, py, dx, dy, groups)
  )
}

const amin = (tensor, axes, keep_dims) => {
  return new Tensor(_amin(tensor._napi_tensor, axes, keep_dims))
}

const amax = (tensor, axes, keep_dims) => {
  return new Tensor(_amax(tensor._napi_tensor, axes, keep_dims))
}

const argmin = (tensor, axis, keep_dims) => {
  return new Tensor(_argmin(tensor._napi_tensor, axis, keep_dims))
}

const argmax = (tensor, axis, keep_dims) => {
  return new Tensor(_argmax(tensor._napi_tensor, axis, keep_dims))
}

const sum = (tensor, axes, keep_dims) => {
  return new Tensor(_sum(tensor._napi_tensor, axes, keep_dims))
}

const cumsum = (tensor, axis) => {
  return new Tensor(_cumsum(tensor._napi_tensor, axis))
}

const mean = (tensor, axes, keep_dims) => {
  return new Tensor(_mean(tensor._napi_tensor, axes, keep_dims))
}

const median = (tensor, axes, keep_dims) => {
  return new Tensor(_median(tensor._napi_tensor, axes, keep_dims))
}

const _var = (tensor, axes, bias, keepdims) => {
  return new Tensor(_var(tensor._napi_tensor, axes, bias, keepdims))
}

const variance = (tensor, axes, bias, keepdims) => {
  return new Tensor(_var(tensor._napi_tensor, axes, bias, keepdims))
}

const std = (tensor, axes, keepdims) => {
  return new Tensor(_std(tensor._napi_tensor, axes, keepdims))
}

const norm = (tensor, axes, p, keepdims) => {
  return new Tensor(_norm(tensor._napi_tensor, axes, p, keepdims))
}

const normalize = (tensor, axes, p, keepdims) => {
  return new Tensor(_norm(tensor._napi_tensor, axes, p, keepdims))
}

const countNonzero = (tensor, axes, keepdims) => {
  return new Tensor(_countNonzero(tensor._napi_tensor, axes, keepdims))
}

const any = (tensor, axes, keepdims) => {
  return new Tensor(_any(tensor._napi_tensor, axes, keepdims))
}

const all = (tensor, axes, keepdims) => {
  return new Tensor(_all(tensor._napi_tensor, axes, keepdims))
}

module.exports = {
  Tensor,
  dtype,
  rand,
  randn,
  full,
  identity,
  ident,
  eye,
  arange,
  iota,
  reshape,
  transpose,
  tile,
  concatenate,
  concat,
  nonzero,
  negative,
  negate,
  logicalNot,
  exp,
  log,
  log1p,
  sin,
  cos,
  sqrt,
  tanh,
  floor,
  ceil,
  rint,
  absolute,
  abs,
  sigmoid,
  erf,
  flip,
  clip,
  roll,
  isnan,
  isinf,
  sign,
  tril,
  triu,
  where,
  sort,
  add,
  sub,
  mul,
  div,
  eq,
  neq,
  lessThan,
  lt,
  lessThanEqual,
  lte,
  greaterThan,
  gt,
  greaterThanEqual,
  gte,
  logicalOr,
  logicalAnd,
  mod,
  bitwiseAnd,
  bitwiseOr,
  bitwiseXor,
  lShift,
  rShift,
  minimum,
  maximum,
  power,
  pow,
  matmul,
  mm,
  conv2d,
  amin,
  amax,
  argmin,
  argmax,
  sum,
  cumsum,
  mean,
  median,
  _var,
  variance,
  std,
  norm,
  normalize,
  countNonzero,
  any,
  all
}
