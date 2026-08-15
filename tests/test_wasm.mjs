import assert from 'node:assert/strict';
import { createFFT } from '../js/ffte.mjs';

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
}

assert.throws(() => fft.c2r1d(new Float64Array(2), 4), RangeError);
assert.throws(() => fft.r2c2d(new Float64Array(4), 2, 3), RangeError);

console.log('All FFTE WebAssembly tests passed');
