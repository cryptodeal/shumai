const {
  init,
  bytesUsed,
  setRowMajor,
  setColMajor,
  isRowMajor,
  isColMajor
} = require('../../build/Release/flashlight_napi_bindings.node')
const { Tensor, dtype } = require('./tensor.cjs')

init()
console.log(isRowMajor()) // true
setColMajor()
console.log(isColMajor()) // true
setRowMajor()
console.log(isRowMajor()) // true

const example = new Tensor(new Float32Array([100, 29, 1, 4]))
console.log(example.elements) // 4
console.log(example.ndim) // 1
console.log(example.dtype) // 1
console.log(example.bytes) // 16
console.log(example.shape64) // BigInt64Array(1) [ 4n ]
console.log(example.shape) // [ 4 ]
console.log(example.toString()) // Tensor[id=XXXXXXXXXX]
console.log(example.toFloat32Array()) // Float32Array(4) [ 100, 29, 1, 4 ]
console.log(example.toFloat64Array()) // Float64Array(4) [ 100, 29, 1, 4 ]
console.log(example.toBoolInt8Array()) // Int8Array(4) [ 1, 1, 1, 1 ]
console.log(example.toInt16Array()) // Int16Array(4) [ 100, 29, 1, 4 ]
console.log(example.toInt32Array()) // Int32Array(4) [ 100, 29, 1, 4 ]
console.log(example.toBigInt64Array()) // BigInt64Array(4) [ 100n, 29n, 1n, 4n ]
console.log(example.toUint8Array()) // Uint8Array(4) [ 100, 29, 1, 4 ]
console.log(example.toUint16Array()) // Uint16Array(4) [ 100, 29, 1, 4 ]
console.log(example.toUint32Array()) // Uint32Array(4) [ 100, 29, 1, 4 ]
console.log(example.toBigUint64Array()) // BigUint64Array(4) [ 100n, 29n, 1n, 4n ]
example.eval()
console.log(bytesUsed()) // 16
const test2 = example.reshape([2, 2])
console.log(test2.shape)
example.dispose()
console.log(bytesUsed()) // 0
console.log(dtype.Float32) // 1
console.log(dtype.Float64) // 2
console.log(dtype.BoolInt8) // 3
console.log(dtype.Int8) // 3
console.log(dtype.Int16) // 4
console.log(dtype.Int32) // 5
console.log(dtype.Int64) // 6
console.log(dtype.BigInt64) // 6
console.log(dtype.Uint8) // 7
console.log(dtype.Uint16) // 8
console.log(dtype.Uint32) // 9
console.log(dtype.Uint64) // 10
console.log(dtype.BigUint64) // 10
