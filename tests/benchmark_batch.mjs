import { performance } from 'node:perf_hooks';
import { createFFT } from '../js/ffte.mjs';

const fft = await createFFT();

function measure(operation, iterations) {
    operation();
    const start = performance.now();
    for (let iteration = 0; iteration < iterations; iteration += 1) {
        operation();
    }
    return (performance.now() - start) / iterations;
}

for (const { length, batchCount, iterations } of [
    { length: 256, batchCount: 128, iterations: 20 },
    { length: 257, batchCount: 128, iterations: 10 },
]) {
    const input = Float64Array.from(
        { length: length * batchCount },
        (_, index) => Math.sin(index * 0.017) + Math.cos(index * 0.031)
    );
    const separate = measure(() => {
        for (let batch = 0; batch < batchCount; batch += 1) {
            fft.r2c1d(input.subarray(batch * length, (batch + 1) * length));
        }
    }, iterations);
    const batched = measure(
        () => fft.r2c1dBatch(input, length),
        iterations
    );

    console.log(
        `${batchCount} x ${length}: separate=${separate.toFixed(2)} ms, ` +
            `batched=${batched.toFixed(2)} ms, ` +
            `speedup=${(separate / batched).toFixed(2)}x`
    );
}
