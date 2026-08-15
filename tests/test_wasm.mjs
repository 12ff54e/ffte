import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createFFT } from '../js/ffte.mjs';

const wasmBytes = await readFile(
    new URL('../dist/ffte.wasm', import.meta.url)
);
const wasmModule = new WebAssembly.Module(wasmBytes);
assert.deepEqual(WebAssembly.Module.imports(wasmModule), [
    {
        module: 'env',
        name: 'emscripten_notify_memory_growth',
        kind: 'function',
    },
]);

const fft = await createFFT();

function almostEqual(actual, expected, tolerance = 2e-9) {
    assert.equal(actual.length, expected.length);
    for (let index = 0; index < actual.length; index += 1) {
        assert.ok(
            Math.abs(actual[index] - expected[index]) < tolerance,
            `index ${index}: expected ${expected[index]}, got ${actual[index]}`
        );
    }
}

function referenceR2c1d(input) {
    const output = new Float64Array(2 * (Math.floor(input.length / 2) + 1));
    for (let frequency = 0; frequency < output.length / 2; frequency += 1) {
        for (let position = 0; position < input.length; position += 1) {
            const angle = -2 * Math.PI * frequency * position / input.length;
            output[2 * frequency] += input[position] * Math.cos(angle);
            output[2 * frequency + 1] += input[position] * Math.sin(angle);
        }
    }
    return output;
}

function referenceR2c2d(input, rows, columns) {
    const halfColumns = Math.floor(columns / 2) + 1;
    const output = new Float64Array(2 * rows * halfColumns);
    for (let outputRow = 0; outputRow < rows; outputRow += 1) {
        for (let outputColumn = 0; outputColumn < halfColumns; outputColumn += 1) {
            const outputIndex = 2 * (outputRow * halfColumns + outputColumn);
            for (let row = 0; row < rows; row += 1) {
                for (let column = 0; column < columns; column += 1) {
                    const angle = -2 * Math.PI * (
                        outputRow * row / rows + outputColumn * column / columns
                    );
                    const value = input[row * columns + column];
                    output[outputIndex] += value * Math.cos(angle);
                    output[outputIndex + 1] += value * Math.sin(angle);
                }
            }
        }
    }
    return output;
}

function referenceC2c1d(input) {
    const length = input.length / 2;
    const output = new Float64Array(input.length);
    for (let frequency = 0; frequency < length; frequency += 1) {
        for (let position = 0; position < length; position += 1) {
            const angle = -2 * Math.PI * frequency * position / length;
            const real = input[2 * position];
            const imaginary = input[2 * position + 1];
            output[2 * frequency] +=
                real * Math.cos(angle) - imaginary * Math.sin(angle);
            output[2 * frequency + 1] +=
                real * Math.sin(angle) + imaginary * Math.cos(angle);
        }
    }
    return output;
}

function referenceC2c2d(input, rows, columns) {
    const output = new Float64Array(input.length);
    for (let outputRow = 0; outputRow < rows; outputRow += 1) {
        for (let outputColumn = 0; outputColumn < columns; outputColumn += 1) {
            const outputIndex = 2 * (outputRow * columns + outputColumn);
            for (let row = 0; row < rows; row += 1) {
                for (let column = 0; column < columns; column += 1) {
                    const inputIndex = 2 * (row * columns + column);
                    const angle = -2 * Math.PI * (
                        outputRow * row / rows + outputColumn * column / columns
                    );
                    const real = input[inputIndex];
                    const imaginary = input[inputIndex + 1];
                    output[outputIndex] +=
                        real * Math.cos(angle) - imaginary * Math.sin(angle);
                    output[outputIndex + 1] +=
                        real * Math.sin(angle) + imaginary * Math.cos(angle);
                }
            }
        }
    }
    return output;
}

for (const length of [
    1, 2, 3, 4, 5, 7, 8, 11, 16, 25, 31, 64, 97, 1000, 1009, 4096,
]) {
    const input = Float64Array.from(
        { length },
        (_, index) => Math.sin(index * 0.37) + 0.2 * Math.cos(index * 1.13)
    );
    const spectrum = fft.r2c1d(input);
    if (length <= 16) almostEqual(spectrum, referenceR2c1d(input));
    almostEqual(fft.c2r1d(spectrum, length), input);

    const batchInput = Float64Array.from(
        { length: length * 3 },
        (_, index) => Math.cos(index * 0.23) + index * 0.0007
    );
    const batchSpectrum = fft.r2c1dBatch(batchInput, length);
    const outputStride = 2 * (Math.floor(length / 2) + 1);
    for (let batch = 0; batch < 3; batch += 1) {
        almostEqual(
            batchSpectrum.slice(batch * outputStride, (batch + 1) * outputStride),
            fft.r2c1d(batchInput.slice(batch * length, (batch + 1) * length))
        );
    }

    const complexInput = Float64Array.from(
        { length: length * 2 },
        (_, index) => Math.sin(index * 0.19) + index * 0.001
    );
    const complexSpectrum = fft.c2c1d(complexInput);
    if (length <= 16) {
        almostEqual(complexSpectrum, referenceC2c1d(complexInput));
    }
    almostEqual(fft.c2c1d(complexSpectrum, true), complexInput);

    const complexBatchInput = Float64Array.from(
        { length: length * 2 * 3 },
        (_, index) => Math.cos(index * 0.11) + index * 0.0002
    );
    const complexBatchSpectrum = fft.c2c1dBatch(complexBatchInput, length);
    const complexStride = length * 2;
    for (let batch = 0; batch < 3; batch += 1) {
        almostEqual(
            complexBatchSpectrum.slice(
                batch * complexStride,
                (batch + 1) * complexStride
            ),
            fft.c2c1d(
                complexBatchInput.slice(
                    batch * complexStride,
                    (batch + 1) * complexStride
                )
            )
        );
    }
    almostEqual(
        fft.c2c1dBatch(complexBatchSpectrum, length, true),
        complexBatchInput
    );
}

for (const [rows, columns] of [
    [1, 5], [3, 2], [3, 5], [4, 4], [5, 7], [13, 17], [16, 32],
]) {
    const input = Float64Array.from(
        { length: rows * columns },
        (_, index) => Math.sin(index * 0.29) + (index % columns) * 0.07
    );
    const spectrum = fft.r2c2d(input, rows, columns);
    if (rows * columns <= 25) {
        almostEqual(spectrum, referenceR2c2d(input, rows, columns));
    }
    almostEqual(fft.c2r2d(spectrum, rows, columns), input);

    const complexInput = Float64Array.from(
        { length: rows * columns * 2 },
        (_, index) => Math.cos(index * 0.17) - index * 0.002
    );
    const complexSpectrum = fft.c2c2d(complexInput, rows, columns);
    if (rows * columns <= 25) {
        almostEqual(
            complexSpectrum,
            referenceC2c2d(complexInput, rows, columns)
        );
    }
    almostEqual(
        fft.c2c2d(complexSpectrum, rows, columns, true),
        complexInput
    );
}

assert.throws(() => fft.c2r1d(new Float64Array(2), 4), RangeError);
assert.throws(() => fft.r2c1dBatch(new Float64Array(5), 2), RangeError);
assert.throws(() => fft.r2c2d(new Float64Array(4), 2, 3), RangeError);

const oldMemory = fft.exports.memory.buffer;
fft.exports.memory.grow(1);
assert.notEqual(fft.exports.memory.buffer, oldMemory);
almostEqual(
    fft.c2r1d(fft.r2c1d(new Float64Array([1, 2, 3, 4, 5])), 5),
    new Float64Array([1, 2, 3, 4, 5])
);

console.log('All FFTE WebAssembly tests passed');
