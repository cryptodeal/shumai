import { existsSync } from 'fs'
import { getStack, Stats, stats } from '../stats'
import { _tidyTracker, cyrb53 } from '../util'

const {
  init,
  bytesUsed: _bytesUsed,
  setRowMajor,
  setColMajor,
  isRowMajor,
  isColMajor,
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
  _variance,
  _std,
  _norm,
  _countNonzero,
  _any,
  _all
} = require('../../build/Release/flashlight_napi_bindings')

init()

export const bytesUsed = _bytesUsed

export const dtype = {
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

export const layout = {
  isRowMajor,
  isColMajor,
  setRowMajor,
  setColMajor
}

export class Tensor {
  private _napi_tensor: any
  private _underlying: ArrayBuffer
  private _ptr: number
  private _deps: Array<Tensor> = []
  private _checkpoint_file: string
  private _checkpoint_callback: () => boolean
  requires_grad = false
  provenance = null
  grad: Tensor = null
  stats: Stats = null
  op = 'constant'

  constructor(obj) {
    if (obj == null) throw new Error('cannot pass `null` to init Tensor')
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

  get deps() {
    return this._deps
  }

  setDeps(deps) {
    this._deps = deps
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
    const i = [this]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('reshape')
    const t = new Tensor(this._napi_tensor.reshape(shape))
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad
    const deps = requires_grad ? [this, shape] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'reshape'
    return t
  }

  astype(dtype) {
    // needs to handle tracking deps/op/provenance/etc...
    return new Tensor(this._napi_tensor.astype(dtype))
  }

  transpose(axes) {
    const i = [this]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('transpose')
    const t = new Tensor(this._napi_tensor.transpose(axes))
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad
    const deps = requires_grad ? [this, axes] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'transpose'
    return t
  }

  tile(shape) {
    const i = [this]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('tile')
    const t = new Tensor(this._napi_tensor.tile(shape))
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad
    const deps = requires_grad ? [this, shape] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'tile'
    return t
  }

  nonzero() {
    const i = [this]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('nonzero')
    const t = new Tensor(this._napi_tensor.nonzero())
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad
    const deps = requires_grad ? [this] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'nonzero'
    return t
  }

  negative() {
    const i = [this]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('negative')
    const t = new Tensor(this._napi_tensor.negative())
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad
    const deps = requires_grad ? [this] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'negative'
    return t
  }

  negate() {
    return new Tensor(this.negative())
  }

  logicalNot() {
    const i = [this]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('logicalNot')
    const t = new Tensor(this._napi_tensor.logicalNot())
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad
    const deps = requires_grad ? [this] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'logicalNot'
    return t
  }

  exp() {
    const i = [this]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('exp')
    const t = new Tensor(this._napi_tensor.exp())
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad
    const deps = requires_grad ? [this] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'exp'
    return t
  }

  log() {
    const i = [this]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('log')
    const t = new Tensor(this._napi_tensor.log())
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad
    const deps = requires_grad ? [this] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'log'
    return t
  }

  log1p() {
    const i = [this]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('log1p')
    const t = new Tensor(this._napi_tensor.log1p())
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad
    const deps = requires_grad ? [this] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'log1p'
    return t
  }

  sin() {
    const i = [this]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('sin')
    const t = new Tensor(this._napi_tensor.sin())
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad
    const deps = requires_grad ? [this] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'sin'
    return t
  }

  cos() {
    const i = [this]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('cos')
    const t = new Tensor(this._napi_tensor.cos())
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad
    const deps = requires_grad ? [this] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'cos'
    return t
  }

  sqrt() {
    const i = [this]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('sqrt')
    const t = new Tensor(this._napi_tensor.sqrt())
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad
    const deps = requires_grad ? [this] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'sqrt'
    return t
  }

  tanh() {
    const i = [this]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('tanh')
    const t = new Tensor(this._napi_tensor.tanh())
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad
    const deps = requires_grad ? [this] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'tanh'
    return t
  }

  floor() {
    const i = [this]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('floor')
    const t = new Tensor(this._napi_tensor.floor())
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad
    const deps = requires_grad ? [this] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'floor'
    return t
  }

  ceil() {
    const i = [this]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('ceil')
    const t = new Tensor(this._napi_tensor.ceil())
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad
    const deps = requires_grad ? [this] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'ceil'
    return t
  }

  rint() {
    const i = [this]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('rint')
    const t = new Tensor(this._napi_tensor.rint())
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad
    const deps = requires_grad ? [this] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'rint'
    return t
  }

  absolute() {
    const i = [this]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('absolute')
    const t = new Tensor(this._napi_tensor.absolute())
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad
    const deps = requires_grad ? [this] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'absolute'
    return t
  }

  abs() {
    return new Tensor(this.absolute())
  }

  sigmoid() {
    const i = [this]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('sigmoid')
    const t = new Tensor(this._napi_tensor.sigmoid())
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad
    const deps = requires_grad ? [this] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'sigmoid'
    return t
  }

  erf() {
    const i = [this]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('erf')
    const t = new Tensor(this._napi_tensor.erf())
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad
    const deps = requires_grad ? [this] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'erf'
    return t
  }

  flip(dim) {
    const i = [this]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('flip')
    const t = new Tensor(
      this._napi_tensor.flip(dim <= 0 ? 0 : dim >= 0xffffffff ? 0xffffffff : +dim || 0)
    )
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad
    const deps = requires_grad
      ? [this, dim <= 0 ? 0 : dim >= 0xffffffff ? 0xffffffff : +dim || 0]
      : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'flip'
    return t
  }

  clip(low, high) {
    const i = [this, low, high]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('clip')
    const t = new Tensor(this._napi_tensor.clip(low._napi_tensor, high._napi_tensor))
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad || low.requires_grad || high.requires_grad
    const deps = requires_grad ? [this, low, high] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance || low.provenance || high.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'clip'
    return t
  }

  roll(shift, axis) {
    const i = [this]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('roll')
    const t = new Tensor(this._napi_tensor.roll(shift | 0, axis | 0))
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad
    const deps = requires_grad ? [this, shift | 0, axis | 0] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'roll'
    return t
  }

  isnan() {
    const i = [this]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('isnan')
    const t = new Tensor(this._napi_tensor.isnan())
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad
    const deps = requires_grad ? [this] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'isnan'
    return t
  }

  isinf() {
    const i = [this]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('isinf')
    const t = new Tensor(this._napi_tensor.isinf())
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad
    const deps = requires_grad ? [this] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'isinf'
    return t
  }

  sign() {
    const i = [this]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('sign')
    const t = new Tensor(this._napi_tensor.sign())
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad
    const deps = requires_grad ? [this] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'sign'
    return t
  }

  tril() {
    const i = [this]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('tril')
    const t = new Tensor(this._napi_tensor.tril())
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad
    const deps = requires_grad ? [this] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'tril'
    return t
  }

  triu() {
    const i = [this]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('triu')
    const t = new Tensor(this._napi_tensor.triu())
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad
    const deps = requires_grad ? [this] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'triu'
    return t
  }

  where(x, y) {
    const i = [this, x, y]
    const ts = i.reduce((s, t) => s || t.stats, void 0)
    const s = ts || stats
    const trace = s.enabled && s.startTrace('where')
    const t = new Tensor(this._napi_tensor.where(x._napi_tensor, y._napi_tensor))
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad || x.requires_grad || y.requires_grad
    const deps = requires_grad ? [this, x, y] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance || x.provenance || y.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'where'
    return t
  }

  sort(dim) {
    const i = [this]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('sort')
    const t = new Tensor(
      this._napi_tensor.sort(dim <= 0 ? 0 : dim >= 0xffffffff ? 0xffffffff : +dim || 0)
    )
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad
    const deps = requires_grad
      ? [this, dim <= 0 ? 0 : dim >= 0xffffffff ? 0xffffffff : +dim || 0]
      : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'sort'
    return t
  }

  add(tensor) {
    const i = [this, tensor]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('add')
    const t = new Tensor(this._napi_tensor.add(tensor._napi_tensor))
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad || tensor.requires_grad
    const deps = requires_grad ? [this, tensor] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance || tensor.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'add'
    return t
  }

  sub(tensor) {
    const i = [this, tensor]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('sub')
    const t = new Tensor(this._napi_tensor.add(tensor._napi_tensor))
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad || tensor.requires_grad
    const deps = requires_grad ? [this, tensor] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance || tensor.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'sub'
    return t
  }

  mul(tensor) {
    const i = [this, tensor]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('mul')
    const t = new Tensor(this._napi_tensor.mul(tensor._napi_tensor))
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad || tensor.requires_grad
    const deps = requires_grad ? [this, tensor] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance || tensor.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'mul'
    return t
  }

  div(tensor) {
    const i = [this, tensor]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('div')
    const t = new Tensor(this._napi_tensor.div(tensor._napi_tensor))
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad || tensor.requires_grad
    const deps = requires_grad ? [this, tensor] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance || tensor.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'div'
    return t
  }

  eq(tensor) {
    const i = [this, tensor]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('eq')
    const t = new Tensor(this._napi_tensor.eq(tensor._napi_tensor))
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad || tensor.requires_grad
    const deps = requires_grad ? [this, tensor] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance || tensor.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'eq'
    return t
  }

  neq(tensor) {
    const i = [this, tensor]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('neq')
    const t = new Tensor(this._napi_tensor.neq(tensor._napi_tensor))
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad || tensor.requires_grad
    const deps = requires_grad ? [this, tensor] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance || tensor.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'neq'
    return t
  }

  lessThan(tensor) {
    const i = [this, tensor]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('lessThan')
    const t = new Tensor(this._napi_tensor.lessThan(tensor._napi_tensor))
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad || tensor.requires_grad
    const deps = requires_grad ? [this, tensor] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance || tensor.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'lessThan'
    return t
  }

  lt(tensor) {
    return new Tensor(this.lessThan(tensor._napi_tensor))
  }

  lessThanEqual(tensor) {
    const i = [this, tensor]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('lessThanEqual')
    const t = new Tensor(this._napi_tensor.lessThanEqual(tensor._napi_tensor))
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad || tensor.requires_grad
    const deps = requires_grad ? [this, tensor] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance || tensor.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'lessThanEqual'
    return t
  }

  lte(tensor) {
    return new Tensor(this.lessThanEqual(tensor._napi_tensor))
  }

  greaterThan(tensor) {
    const i = [this, tensor]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('greaterThan')
    const t = new Tensor(this._napi_tensor.greaterThan(tensor._napi_tensor))
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad || tensor.requires_grad
    const deps = requires_grad ? [this, tensor] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance || tensor.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'greaterThan'
    return t
  }

  gt(tensor) {
    return new Tensor(this.greaterThan(tensor._napi_tensor))
  }

  greaterThanEqual(tensor) {
    const i = [this, tensor]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('greaterThanEqual')
    const t = new Tensor(this._napi_tensor.greaterThanEqual(tensor._napi_tensor))
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad || tensor.requires_grad
    const deps = requires_grad ? [this, tensor] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance || tensor.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'greaterThanEqual'
    return t
  }

  gte(tensor) {
    return new Tensor(this.greaterThanEqual(tensor._napi_tensor))
  }

  logicalOr(tensor) {
    const i = [this, tensor]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('logicalOr')
    const t = new Tensor(this._napi_tensor.logicalOr(tensor._napi_tensor))
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad || tensor.requires_grad
    const deps = requires_grad ? [this, tensor] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance || tensor.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'logicalOr'
    return t
  }

  logicalAnd(tensor) {
    const i = [this, tensor]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('logicalAnd')
    const t = new Tensor(this._napi_tensor.logicalAnd(tensor._napi_tensor))
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad || tensor.requires_grad
    const deps = requires_grad ? [this, tensor] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance || tensor.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'logicalAnd'
    return t
  }

  mod(tensor) {
    const i = [this, tensor]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('mod')
    const t = new Tensor(this._napi_tensor.mod(tensor._napi_tensor))
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad || tensor.requires_grad
    const deps = requires_grad ? [this, tensor] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance || tensor.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'mod'
    return t
  }

  bitwiseAnd(tensor) {
    const i = [this, tensor]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('bitwiseAnd')
    const t = new Tensor(this._napi_tensor.bitwiseAnd(tensor._napi_tensor))
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad || tensor.requires_grad
    const deps = requires_grad ? [this, tensor] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance || tensor.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'bitwiseAnd'
    return t
  }

  bitwiseOr(tensor) {
    const i = [this, tensor]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('bitwiseOr')
    const t = new Tensor(this._napi_tensor.bitwiseOr(tensor._napi_tensor))
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad || tensor.requires_grad
    const deps = requires_grad ? [this, tensor] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance || tensor.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'bitwiseOr'
    return t
  }

  bitwiseXor(tensor) {
    const i = [this, tensor]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('bitwiseXor')
    const t = new Tensor(this._napi_tensor.bitwiseXor(tensor._napi_tensor))
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad || tensor.requires_grad
    const deps = requires_grad ? [this, tensor] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance || tensor.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'bitwiseXor'
    return t
  }

  lShift(tensor) {
    const i = [this, tensor]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('lShift')
    const t = new Tensor(this._napi_tensor.lShift(tensor._napi_tensor))
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad || tensor.requires_grad
    const deps = requires_grad ? [this, tensor] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance || tensor.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'lShift'
    return t
  }

  rShift(tensor) {
    const i = [this, tensor]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('rShift')
    const t = new Tensor(this._napi_tensor.rShift(tensor._napi_tensor))
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad || tensor.requires_grad
    const deps = requires_grad ? [this, tensor] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance || tensor.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'rShift'
    return t
  }

  minimum(tensor) {
    const i = [this, tensor]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('minimum')
    const t = new Tensor(this._napi_tensor.minimum(tensor._napi_tensor))
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad || tensor.requires_grad
    const deps = requires_grad ? [this, tensor] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance || tensor.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'minimum'
    return t
  }

  maximum(tensor) {
    const i = [this, tensor]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('maximum')
    const t = new Tensor(this._napi_tensor.maximum(tensor._napi_tensor))
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad || tensor.requires_grad
    const deps = requires_grad ? [this, tensor] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance || tensor.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'maximum'
    return t
  }

  power(tensor) {
    const i = [this, tensor]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('power')
    const t = new Tensor(this._napi_tensor.power(tensor._napi_tensor))
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad || tensor.requires_grad
    const deps = requires_grad ? [this, tensor] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance || tensor.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'power'
    return t
  }

  pow(tensor) {
    return new Tensor(this.power(tensor._napi_tensor))
  }

  matmul(tensor) {
    const i = [this, tensor]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('matmul')
    const t = new Tensor(this._napi_tensor.matmul(tensor._napi_tensor))
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad || tensor.requires_grad
    const deps = requires_grad ? [this, tensor] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance || tensor.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'matmul'
    return t
  }

  mm(tensor) {
    return new Tensor(this.matmul(tensor._napi_tensor))
  }

  conv2d(weights, sx = 1, sy = 1, px = 0, py = 0, dx = 1, dy = 1, groups = 1) {
    const i = [this, weights]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('conv2d')
    const t = new Tensor(
      this._napi_tensor.conv2d(
        weights._napi_tensor,
        sx | 0,
        sy | 0,
        px | 0,
        py | 0,
        dx | 0,
        dy | 0,
        groups | 0
      )
    )
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad || weights.requires_grad
    const deps = requires_grad
      ? [this, weights, sx | 0, sy | 0, px | 0, py | 0, dx | 0, dy | 0, groups | 0]
      : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance || weights.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'conv2d'
    return t
  }

  amin(axes = [], keep_dims = false) {
    const i = [this]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('amin')
    const t = new Tensor(this._napi_tensor.amin(axes, !!keep_dims))
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad
    const deps = requires_grad ? [this, axes, !!keep_dims] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'amin'
    return t
  }

  amax(axes = [], keep_dims = false) {
    const i = [this]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('amax')
    const t = new Tensor(this._napi_tensor.amax(axes, !!keep_dims))
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad
    const deps = requires_grad ? [this, axes, !!keep_dims] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'amax'
    return t
  }

  argmin(axis, keep_dims = false) {
    const i = [this]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('argmin')
    const t = new Tensor(this._napi_tensor.argmin(axis | 0, !!keep_dims))
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad
    const deps = requires_grad ? [this, axis | 0, !!keep_dims] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'argmin'
    return t
  }

  argmax(axis, keep_dims = false) {
    const i = [this]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('argmax')
    const t = new Tensor(this._napi_tensor.argmax(axis | 0, !!keep_dims))
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad
    const deps = requires_grad ? [this, axis | 0, !!keep_dims] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'argmax'
    return t
  }

  sum(axes = [], keep_dims = false) {
    const i = [this]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('sum')
    const t = new Tensor(this._napi_tensor.sum(axes, !!keep_dims))
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad
    const deps = requires_grad ? [this, axes, !!keep_dims] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'sum'
    return t
  }

  cumsum(axis) {
    const i = [this]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('cumsum')
    const t = new Tensor(this._napi_tensor.cumsum(axis | 0))
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad
    const deps = requires_grad ? [this, axis | 0] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'cumsum'
    return t
  }

  mean(axes = [], keep_dims = false) {
    const i = [this]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('mean')
    const t = new Tensor(this._napi_tensor.mean(axes, !!keep_dims))
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad
    const deps = requires_grad ? [this, axes, !!keep_dims] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'mean'
    return t
  }

  median(axes = [], keep_dims = false) {
    const i = [this]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('median')
    const t = new Tensor(this._napi_tensor.mean(axes, !!keep_dims))
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad
    const deps = requires_grad ? [this, axes, !!keep_dims] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'median'
    return t
  }

  _var(axes = [], bias = false, keep_dims = false) {
    const i = [this]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('var')
    const t = new Tensor(this._napi_tensor._var(axes, !!bias, !!keep_dims))
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad
    const deps = requires_grad ? [this, axes, !!bias, !!keep_dims] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'var'
    return t
  }

  variance(axes = [], bias = false, keep_dims = false) {
    return new Tensor(this._var(axes, bias, keep_dims))
  }

  std(axes = [], keep_dims = false) {
    const i = [this]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('std')
    const t = new Tensor(this._napi_tensor.std(axes, keep_dims))
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad
    const deps = requires_grad ? [this, axes, !!keep_dims] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'std'
    return t
  }

  norm(axes = [], p = 2, keep_dims = false) {
    const i = [this]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('norm')
    const t = new Tensor(
      this._napi_tensor.norm(axes, p + 0.00000000000001 - 0.00000000000001, !!keep_dims)
    )
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad
    const deps = requires_grad
      ? [this, axes, p + 0.00000000000001 - 0.00000000000001, !!keep_dims]
      : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'norm'
    return t
  }

  normalize(axes = [], p = 2, keep_dims = false) {
    return this.norm(axes, p, keep_dims)
  }

  countNonzero(axes = [], keep_dims = false) {
    const i = [this]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('countNonzero')
    const t = new Tensor(this._napi_tensor.countNonzero(axes, !!keep_dims))
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad
    const deps = requires_grad ? [this, axes, !!keep_dims] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'countNonzero'
    return t
  }

  any(axes = [], keep_dims = false) {
    const i = [this]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('any')
    const t = new Tensor(this._napi_tensor.any(axes, !!keep_dims))
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad
    const deps = requires_grad ? [this, axes, !!keep_dims] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'any'
    return t
  }

  all(axes = [], keep_dims = false) {
    const i = [this]
    const ts = <Stats>i.reduce((s, t) => s || t.stats, void 0)
    const s = <Stats>(ts || stats)
    const trace = s.enabled && s.startTrace('all')
    const t = new Tensor(this._napi_tensor.all(axes, !!keep_dims))
    trace && s.stopTrace(trace)

    const requires_grad = this.requires_grad
    const deps = requires_grad ? [this, axes, !!keep_dims] : []
    t.setDeps(deps)
    t.stats = ts
    t.provenance = this.provenance
    t.requires_grad = requires_grad

    trace && s.logTrace(trace, i, t)

    t.op = 'all'
    return t
  }
}

export const rand = (shape) => {
  const i = []
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('rand')
  const t = new Tensor(_rand(shape))

  trace && s.stopTrace(trace)

  const requires_grad = false
  const deps = requires_grad ? [shape] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'rand'
  return t
}

export const randn = (shape) => {
  const i = []
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('randn')
  const t = new Tensor(_randn(shape))

  trace && s.stopTrace(trace)

  const requires_grad = false
  const deps = requires_grad ? [shape] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = t.requires_grad = requires_grad
  trace && s.logTrace(trace, i, t)

  t.op = 'randn'
  return t
}

export const full = (shape, val) => {
  const i = []
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('full')
  const t = new Tensor(_full(shape, Math.fround(val)))
  trace && s.stopTrace(trace)

  const requires_grad = false
  const deps = requires_grad ? [shape, Math.fround(val)] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)
  t.op = 'full'
  return t
}

export const identity = (dim) => {
  const i = []
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('identity')
  const t = new Tensor(_identity(typeof dim === 'bigint' ? dim : BigInt(dim || 0)))
  trace && s.stopTrace(trace)

  const requires_grad = false
  const deps = requires_grad ? [dim.constructor === BigInt ? dim : BigInt(dim || 0)] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'identity'
  return t
}

export const ident = (dim) => {
  return new Tensor(identity(dim))
}

export const eye = (dim) => {
  return new Tensor(identity(dim))
}

export const arange = (start, end, step = 1) => {
  const i = []
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('arange')
  const t = new Tensor(_arange(Math.fround(start), Math.fround(end), Math.fround(step)))
  trace && s.stopTrace(trace)

  const requires_grad = false
  const deps = requires_grad ? [Math.fround(start), Math.fround(end), Math.fround(step)] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'arange'
  return t
}

export const iota = (dims, tileDims = [1]) => {
  const i = []
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('iota')
  const t = new Tensor(_iota(dims, tileDims))
  trace && s.stopTrace(trace)

  const requires_grad = false
  const deps = requires_grad ? [dims, tileDims] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'iota'
  return t
}

export const reshape = (tensor, shape) => {
  const i = [tensor]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('reshape')
  const t = new Tensor(_reshape(tensor._napi_tensor, shape))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad
  const deps = requires_grad ? [tensor, shape] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'reshape'
  return t
}

export const transpose = (tensor, axes) => {
  const i = [tensor]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('transpose')
  const t = new Tensor(_transpose(tensor._napi_tensor, axes))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad
  const deps = requires_grad ? [tensor, axes] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'transpose'
  return t
}

export const tile = (tensor, shape) => {
  const i = [tensor]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('tile')
  const t = new Tensor(_tile(tensor._napi_tensor, shape))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad
  const deps = requires_grad ? [tensor, shape] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'tile'
  return t
}

export const concatenate = (tensors, axis) => {
  if (axis < 0) {
    for (let i = 0; i < tensors.length; ++i) {
      if (tensors[i].shape.length === 0) {
        tensors[i] = tensors[i].reshape([1])
      }
    }
  }
  const i = []
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('concatenate')
  const t = new Tensor(
    _concatenate(
      tensors.map((t) => t._napi_tensor),
      axis | 0
    )
  )
  trace && s.stopTrace(trace)

  const requires_grad = tensors.reduce((r, c) => r || c.requires_grad, false)
  const deps = requires_grad ? [...tensors, axis | 0] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensors.reduce((r, c) => r || c.provenance, 0)
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'concatenate'
  return t
}

export const concat = (tensors, axis) => {
  return new Tensor(
    concat(
      tensors.map((t) => t._napi_tensor),
      axis
    )
  )
}

export const nonzero = (tensor) => {
  const i = [tensor]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('nonzero')
  const t = new Tensor(_nonzero(tensor._napi_tensor))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad
  const deps = requires_grad ? [tensor] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'nonzero'
  return t
}

export const negative = (tensor) => {
  const i = [tensor]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('negative')
  const t = new Tensor(_negative(tensor._napi_tensor))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad
  const deps = requires_grad ? [tensor] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'negative'
  return t
}

export const negate = (tensor) => {
  return new Tensor(negative(tensor._napi_tensor))
}

export const logicalNot = (tensor) => {
  const i = [tensor]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('logicalNot')
  const t = new Tensor(_logicalNot(tensor._napi_tensor))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad
  const deps = requires_grad ? [tensor] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'logicalNot'
  return t
}

export const exp = (tensor) => {
  const i = [tensor]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('exp')
  const t = new Tensor(_exp(tensor._napi_tensor))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad
  const deps = requires_grad ? [tensor] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'exp'
  return t
}

export const log = (tensor) => {
  const i = [tensor]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('log')
  const t = new Tensor(_log(tensor._napi_tensor))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad
  const deps = requires_grad ? [tensor] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'log'
  return t
}

export const log1p = (tensor) => {
  const i = [tensor]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('log1p')
  const t = new Tensor(_log1p(tensor._napi_tensor))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad
  const deps = requires_grad ? [tensor] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'log1p'
  return t
}

export const sin = (tensor) => {
  const i = [tensor]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('sin')
  const t = new Tensor(_sin(tensor._napi_tensor))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad
  const deps = requires_grad ? [tensor] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'sin'
  return t
}

export const cos = (tensor) => {
  const i = [tensor]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('cos')
  const t = new Tensor(_cos(tensor._napi_tensor))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad
  const deps = requires_grad ? [tensor] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'cos'
  return t
}

export const sqrt = (tensor) => {
  const i = [tensor]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('sqrt')
  const t = new Tensor(_sqrt(tensor._napi_tensor))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad
  const deps = requires_grad ? [tensor] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'sqrt'
  return t
}

export const tanh = (tensor) => {
  const i = [tensor]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('tanh')
  const t = new Tensor(_tanh(tensor._napi_tensor))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad
  const deps = requires_grad ? [tensor] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'tanh'
  return t
}

export const floor = (tensor) => {
  const i = [tensor]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('floor')
  const t = new Tensor(_floor(tensor._napi_tensor))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad
  const deps = requires_grad ? [tensor] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'floor'
  return t
}

export const ceil = (tensor) => {
  const i = [tensor]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('ceil')
  const t = new Tensor(_ceil(tensor._napi_tensor))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad
  const deps = requires_grad ? [tensor] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'ceil'
  return t
}

export const rint = (tensor) => {
  const i = [tensor]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('rint')
  const t = new Tensor(_rint(tensor._napi_tensor))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad
  const deps = requires_grad ? [tensor] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'rint'
  return t
}

export const absolute = (tensor) => {
  const i = [tensor]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('absolute')
  const t = new Tensor(_absolute(tensor._napi_tensor))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad
  const deps = requires_grad ? [tensor] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'absolute'
  return t
}

export const abs = (tensor) => {
  return new Tensor(absolute(tensor._napi_tensor))
}

export const sigmoid = (tensor) => {
  const i = [tensor]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('sigmoid')
  const t = new Tensor(_sigmoid(tensor._napi_tensor))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad
  const deps = requires_grad ? [tensor] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'sigmoid'
  return t
}

export const erf = (tensor) => {
  const i = [tensor]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('erf')
  const t = new Tensor(_erf(tensor._napi_tensor))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad
  const deps = requires_grad ? [tensor] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'erf'
  return t
}

export const flip = (tensor, dim) => {
  const i = [tensor]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('flip')
  const t = new Tensor(
    _flip(tensor._napi_tensor, dim <= 0 ? 0 : dim >= 0xffffffff ? 0xffffffff : +dim || 0)
  )
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad
  const deps = requires_grad
    ? [tensor, dim <= 0 ? 0 : dim >= 0xffffffff ? 0xffffffff : +dim || 0]
    : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'flip'
  return t
}

export const clip = (tensor, low, high) => {
  const i = [tensor, low, high]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('clip')
  const t = new Tensor(_clip(tensor._napi_tensor, low._napi_tensor, high._napi_tensor))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad || low.requires_grad || high.requires_grad
  const deps = requires_grad ? [tensor, low, high] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance || low.provenance || high.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'clip'
  return t
}

export const roll = (tensor, shift, axis) => {
  const i = [tensor]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('roll')
  const t = new Tensor(_roll(tensor._napi_tensor, shift | 0, axis | 0))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad
  const deps = requires_grad ? [tensor, shift | 0, axis | 0] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'roll'
  return t
}

export const isnan = (tensor) => {
  const i = [tensor]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('isnan')
  const t = new Tensor(_isnan(tensor._napi_tensor))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad
  const deps = requires_grad ? [tensor] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'isnan'
  return t
}

export const isinf = (tensor) => {
  const i = [tensor]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('isinf')
  const t = new Tensor(_isinf(tensor._napi_tensor))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad
  const deps = requires_grad ? [tensor] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'isinf'
  return t
}

export const sign = (tensor) => {
  const i = [tensor]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('sign')
  const t = new Tensor(_sign(tensor._napi_tensor))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad
  const deps = requires_grad ? [tensor] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'sign'
  return t
}

export const tril = (tensor) => {
  const i = [tensor]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('tril')
  const t = new Tensor(_tril(tensor._napi_tensor))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad
  const deps = requires_grad ? [tensor] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'tril'
  return t
}

export const triu = (tensor) => {
  const i = [tensor]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('triu')
  const t = new Tensor(_triu(tensor._napi_tensor))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad
  const deps = requires_grad ? [tensor] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'triu'
  return t
}

export const where = (cond, x, y) => {
  const i = [cond, x, y]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('where')
  const t = new Tensor(_where(cond._napi_tensor, x._napi_tensor, y._napi_tensor))
  trace && s.stopTrace(trace)

  const requires_grad = cond.requires_grad || x.requires_grad || y.requires_grad
  const deps = requires_grad ? [cond, x, y] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = cond.provenance || x.provenance || y.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'where'
  return t
}

export const sort = (tensor, dim) => {
  const i = [tensor]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('sort')
  const t = new Tensor(
    _sort(tensor._napi_tensor, dim <= 0 ? 0 : dim >= 0xffffffff ? 0xffffffff : +dim || 0)
  )
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad
  const deps = requires_grad
    ? [tensor, dim <= 0 ? 0 : dim >= 0xffffffff ? 0xffffffff : +dim || 0]
    : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'sort'
  return t
}

export const add = (tensor, other) => {
  const i = [tensor, other]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('add')
  const t = new Tensor(_add(tensor._napi_tensor, other._napi_tensor))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad || other.requires_grad
  const deps = requires_grad ? [tensor, other] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance || other.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'add'
  return t
}

export const sub = (tensor, other) => {
  const i = [tensor, other]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('sub')
  const t = new Tensor(_sub(tensor._napi_tensor, other._napi_tensor))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad || other.requires_grad
  const deps = requires_grad ? [tensor, other] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance || other.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'sub'
  return t
}

export const mul = (tensor, other) => {
  const i = [tensor, other]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('mul')
  const t = new Tensor(_mul(tensor._napi_tensor, other._napi_tensor))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad || other.requires_grad
  const deps = requires_grad ? [tensor, other] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance || other.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'mul'
  return t
}

export const div = (tensor, other) => {
  const i = [tensor, other]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('div')
  const t = new Tensor(_div(tensor._napi_tensor, other._napi_tensor))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad || other.requires_grad
  const deps = requires_grad ? [tensor, other] : []
  t.stats = ts
  t.provenance = tensor.provenance || other.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'div'
  return t
}

export const eq = (tensor, other) => {
  const i = [tensor, other]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('eq')
  const t = new Tensor(_eq(tensor._napi_tensor, other._napi_tensor))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad || other.requires_grad
  const deps = requires_grad ? [tensor, other] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance || other.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'eq'
  return t
}

export const neq = (tensor, other) => {
  const i = [tensor, other]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('neq')
  const t = new Tensor(_neq(tensor._napi_tensor, other._napi_tensor))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad || other.requires_grad
  const deps = requires_grad ? [tensor, other] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance || other.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'neq'
  return t
}

export const lessThan = (tensor, other) => {
  const i = [tensor, other]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('lessThan')
  const t = new Tensor(_lessThan(tensor._napi_tensor, other._napi_tensor))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad || other.requires_grad
  const deps = requires_grad ? [tensor, other] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance || other.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'lessThan'
  return t
}

export const lt = (tensor, other) => {
  return new Tensor(lessThan(tensor._napi_tensor, other._napi_tensor))
}

export const lessThanEqual = (tensor, other) => {
  const i = [tensor, other]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('lessThanEqual')
  const t = new Tensor(_lessThanEqual(tensor._napi_tensor, other._napi_tensor))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad || other.requires_grad
  const deps = requires_grad ? [tensor, other] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance || other.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'lessThanEqual'
  return t
}

export const lte = (tensor, other) => {
  return new Tensor(lessThanEqual(tensor._napi_tensor, other._napi_tensor))
}

export const greaterThan = (tensor, other) => {
  const i = [tensor, other]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('greaterThan')
  const t = new Tensor(_greaterThan(tensor._napi_tensor, other._napi_tensor))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad || other.requires_grad
  const deps = requires_grad ? [tensor, other] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance || other.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'greaterThan'
  return t
}

export const gt = (tensor, other) => {
  return new Tensor(greaterThan(tensor._napi_tensor, other._napi_tensor))
}

export const greaterThanEqual = (tensor, other) => {
  const i = [tensor, other]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('greaterThanEqual')
  const t = new Tensor(_greaterThanEqual(tensor._napi_tensor, other._napi_tensor))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad || other.requires_grad
  const deps = requires_grad ? [tensor, other] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance || other.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'greaterThanEqual'
  return t
}

export const gte = (tensor, other) => {
  return new Tensor(greaterThanEqual(tensor._napi_tensor, other._napi_tensor))
}

export const logicalOr = (tensor, other) => {
  const i = [tensor, other]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('logicalOr')
  const t = new Tensor(_logicalOr(tensor._napi_tensor, other._napi_tensor))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad || other.requires_grad
  const deps = requires_grad ? [tensor, other] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance || other.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'logicalOr'
  return t
}

export const logicalAnd = (tensor, other) => {
  const i = [tensor, other]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('logicalAnd')
  const t = new Tensor(_logicalAnd(tensor._napi_tensor, other._napi_tensor))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad || other.requires_grad
  const deps = requires_grad ? [tensor, other] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance || other.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'logicalAnd'
  return t
}

export const mod = (tensor, other) => {
  const i = [tensor, other]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('mod')
  const t = new Tensor(_mod(tensor._napi_tensor, other._napi_tensor))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad || other.requires_grad
  const deps = requires_grad ? [tensor, other] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance || other.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'mod'
  return t
}

export const bitwiseAnd = (tensor, other) => {
  const i = [tensor, other]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('bitwiseAnd')
  const t = new Tensor(_bitwiseAnd(tensor._napi_tensor, other._napi_tensor))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad || other.requires_grad
  const deps = requires_grad ? [tensor, other] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance || other.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'bitwiseAnd'
  return t
}

export const bitwiseOr = (tensor, other) => {
  const i = [tensor, other]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('bitwiseOr')
  const t = new Tensor(_bitwiseOr(tensor._napi_tensor, other._napi_tensor))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad || other.requires_grad
  const deps = requires_grad ? [tensor, other] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance || other.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'bitwiseOr'
  return t
}

export const bitwiseXor = (tensor, other) => {
  const i = [tensor, other]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('bitwiseXor')
  const t = new Tensor(_bitwiseXor(tensor._napi_tensor, other._napi_tensor))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad || other.requires_grad
  const deps = requires_grad ? [tensor, other] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance || other.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'bitwiseXor'
  return t
}

export const lShift = (tensor, other) => {
  const i = [tensor, other]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('lShift')
  const t = new Tensor(_lShift(tensor._napi_tensor, other._napi_tensor))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad || other.requires_grad
  const deps = requires_grad ? [tensor, other] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance || other.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'lShift'
  return t
}

export const rShift = (tensor, other) => {
  const i = [tensor, other]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('rShift')
  const t = new Tensor(_rShift(tensor._napi_tensor, other._napi_tensor))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad || other.requires_grad
  const deps = requires_grad ? [tensor, other] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance || other.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'rShift'
  return t
}

export const minimum = (tensor, other) => {
  const i = [tensor, other]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('minimum')
  const t = new Tensor(_minimum(tensor._napi_tensor, other._napi_tensor))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad || other.requires_grad
  const deps = requires_grad ? [tensor, other] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance || other.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'minimum'
  return t
}

export const maximum = (tensor, other) => {
  const i = [tensor, other]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('maximum')
  const t = new Tensor(_maximum(tensor._napi_tensor, other._napi_tensor))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad || other.requires_grad
  const deps = requires_grad ? [tensor, other] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance || other.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'maximum'
  return t
}

export const power = (tensor, other) => {
  const i = [tensor, other]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('power')
  const t = new Tensor(_power(tensor._napi_tensor, other._napi_tensor))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad || other.requires_grad
  const deps = requires_grad ? [tensor, other] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance || other.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'power'
  return t
}

export const pow = (tensor, other) => {
  return new Tensor(power(tensor._napi_tensor, other._napi_tensor))
}

export const matmul = (tensor, other) => {
  const i = [tensor, other]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('matmul')
  const t = new Tensor(_matmul(tensor._napi_tensor, other._napi_tensor))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad || other.requires_grad
  const deps = requires_grad ? [tensor, other] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance || other.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'matmul'
  return t
}

export const mm = (tensor, other) => {
  return new Tensor(matmul(tensor._napi_tensor, other._napi_tensor))
}

const conv2d = (tensor, weights, sx = 1, sy = 1, px = 0, py = 0, dx = 1, dy = 1, groups = 1) => {
  const i = [tensor, weights]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('conv2d')
  const t = new Tensor(
    _conv2d(
      tensor._napi_tensor,
      weights._napi_tensor,
      sx | 0,
      sy | 0,
      px | 0,
      py | 0,
      dx | 0,
      dy | 0,
      groups | 0
    )
  )
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad || weights.requires_grad
  const deps = requires_grad
    ? [tensor, weights, sx | 0, sy | 0, px | 0, py | 0, dx | 0, dy | 0, groups | 0]
    : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance || weights.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'conv2d'
  return t
}

export const amin = (tensor, axes = [], keep_dims = false) => {
  const i = [tensor]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('amin')
  const t = new Tensor(_amin(tensor._napi_tensor, axes, keep_dims))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad
  const deps = requires_grad ? [tensor, axes, !!keep_dims] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'amin'
  return t
}

export const amax = (tensor, axes = [], keep_dims = false) => {
  const i = [tensor]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('amax')
  const t = new Tensor(_amax(tensor._napi_tensor, axes, keep_dims))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad
  const deps = requires_grad ? [tensor, axes, !!keep_dims] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'amax'
  return t
}

export const argmin = (tensor, axis, keep_dims = false) => {
  const i = [tensor]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('argmin')
  const t = new Tensor(_argmin(tensor._napi_tensor, axis | 0, !!keep_dims))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad
  const deps = requires_grad ? [tensor, axis | 0, !!keep_dims] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'argmin'
  return t
}

export const argmax = (tensor, axis, keep_dims = false) => {
  const i = [tensor]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('argmax')
  const t = new Tensor(_argmax(tensor._napi_tensor, axis | 0, !!keep_dims))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad
  const deps = requires_grad ? [tensor, axis | 0, !!keep_dims] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'argmax'
  return t
}

export const sum = (tensor, axes, keep_dims = false) => {
  const i = [tensor]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('sum')
  const t = new Tensor(_sum(tensor._napi_tensor, axes, !!keep_dims))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad
  const deps = requires_grad ? [tensor, axes, !!keep_dims] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'sum'
  return t
}

export const cumsum = (tensor, axis) => {
  const i = [tensor]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('cumsum')
  const t = new Tensor(_cumsum(tensor._napi_tensor, axis | 0))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad
  const deps = requires_grad ? [tensor, axis | 0] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'cumsum'
  return t
}

export const mean = (tensor, axes = [], keep_dims = false) => {
  const i = [tensor]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('mean')
  const t = new Tensor(_mean(tensor._napi_tensor, axes, !!keep_dims))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad
  const deps = requires_grad ? [tensor, axes, !!keep_dims] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'mean'
  return t
}

export const median = (tensor, axes = [], keep_dims = false) => {
  const i = [tensor]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('median')
  const t = new Tensor(_median(tensor._napi_tensor, axes, !!keep_dims))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad
  const deps = requires_grad ? [tensor, axes, !!keep_dims] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'median'
  return t
}

export const _var = (tensor, axes = [], bias = false, keep_dims = false) => {
  const i = [tensor]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('var')
  const t = new Tensor(_variance(tensor._napi_tensor, axes, !!bias, !!keep_dims))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad
  const deps = requires_grad ? [tensor, axes, !!bias, !!keep_dims] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'var'
  return t
}

export const variance = (tensor, axes = [], bias = false, keep_dims = false) => {
  return new Tensor(_var(tensor._napi_tensor, axes, bias, keep_dims))
}

export const std = (tensor, axes = [], keep_dims = false) => {
  const i = [tensor]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('std')
  const t = new Tensor(_std(tensor._napi_tensor, axes, !!keep_dims))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad
  const deps = requires_grad ? [tensor, axes, !!keep_dims] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'std'
  return t
}

export const norm = (tensor, axes = [], p = 2, keep_dims = false) => {
  const i = [tensor]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('norm')
  const t = new Tensor(
    _norm(tensor._napi_tensor, axes, p + 0.00000000000001 - 0.00000000000001, !!keep_dims)
  )
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad
  const deps = requires_grad
    ? [tensor, axes, p + 0.00000000000001 - 0.00000000000001, !!keep_dims]
    : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'norm'
  return t
}

export const normalize = (tensor, axes = [], p = 2, keep_dims = false) => {
  return new Tensor(norm(tensor._napi_tensor, axes, p, keep_dims))
}

export const countNonzero = (tensor, axes = [], keep_dims = false) => {
  const i = [tensor]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('countNonzero')
  const t = new Tensor(_countNonzero(tensor._napi_tensor, axes, !!keep_dims))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad
  const deps = requires_grad ? [tensor, axes, !!keep_dims] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'countNonzero'
  return t
}

export const any = (tensor, axes = [], keep_dims = false) => {
  const i = [tensor]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('any')
  const t = new Tensor(_any(tensor._napi_tensor, axes, !!keep_dims))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad
  const deps = requires_grad ? [tensor, axes, !!keep_dims] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'any'
  return t
}

export const all = (tensor, axes = [], keep_dims = false) => {
  const i = [tensor]
  const ts = i.reduce((s, t) => s || t.stats, void 0)
  const s = ts || stats
  const trace = s.enabled && s.startTrace('all')
  const t = new Tensor(_all(tensor._napi_tensor, axes, !!keep_dims))
  trace && s.stopTrace(trace)

  const requires_grad = tensor.requires_grad
  const deps = requires_grad ? [tensor, axes, !!keep_dims] : []
  t.setDeps(deps)
  t.stats = ts
  t.provenance = tensor.provenance
  t.requires_grad = requires_grad

  trace && s.logTrace(trace, i, t)

  t.op = 'all'
  return t
}
