import * as sm from '@shumai/shumai'

const t0 = performance.now()
sm.layout.isRowMajor()
sm.layout.setColMajor()
sm.layout.isColMajor()
sm.layout.setRowMajor()
sm.layout.isRowMajor()

const example = new sm.Tensor(new Float32Array([100, 29, 1, 4]))

example.elements // 4
example.ndim // 1
example.dtype // 1
// example.bytes) // 16
example.shape64 // BigInt64Array(1) [ 4n ]
example.shape // [ 4 ]
example.toString() // Tensor[id=XXXXXXXXXX]
example.toFloat32Array() // Float32Array(4) [ 100, 29, 1, 4 ]
example.toFloat64Array() // Float64Array(4) [ 100, 29, 1, 4 ]
// example.toBoolInt8Array()) // Int8Array(4) [ 1, 1, 1, 1 ]
example.toInt16Array() // Int16Array(4) [ 100, 29, 1, 4 ]
example.toInt32Array() // Int32Array(4) [ 100, 29, 1, 4 ]
example.toBigInt64Array() // BigInt64Array(4) [ 100n, 29n, 1n, 4n ]
example.toUint8Array() // Uint8Array(4) [ 100, 29, 1, 4 ]
example.toUint16Array() // Uint16Array(4) [ 100, 29, 1, 4 ]
example.toUint32Array() // Uint32Array(4) [ 100, 29, 1, 4 ]
example.toBigUint64Array() // BigUint64Array(4) [ 100n, 29n, 1n, 4n ]
example.eval()
sm.bytesUsed() // 16
// TODO: fix, throws AF error
// example.reshape([2, 2])
// example.shape)
example.dispose()
sm.bytesUsed() // 0
sm.dtype.Float32 // 1
sm.dtype.Float64 // 2
sm.dtype.BoolInt8 // 3
sm.dtype.BoolInt8 // 3
sm.dtype.Int16 // 4
sm.dtype.Int32 // 5
sm.dtype.Int64 // 6
sm.dtype.BigInt64 // 6
sm.dtype.Uint8 // 7
sm.dtype.Uint16 // 8
sm.dtype.Uint32 // 9
sm.dtype.Uint64 // 10
sm.dtype.BigUint64 // 10
const t1 = performance.now()
console.log(`Napi tensor took ${t1 - t0} milliseconds.`)
