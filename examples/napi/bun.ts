import { napi } from '@shumai/shumai'
const { Tensor } = napi

const test = new Tensor(new Float32Array([1, 2, 3])).reshape([3, 1])
console.log(test.shape)
