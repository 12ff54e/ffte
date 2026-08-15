import createFFTE from '../dist/ffte.js';

const SUCCESS = 0;

function requirePositiveInteger(value, name) {
    if (!Number.isSafeInteger(value) || value <= 0) {
        throw new RangeError(`${name} must be a positive safe integer`);
    }
}

function checkedProduct(left, right, name) {
    const result = left * right;
    if (!Number.isSafeInteger(result)) {
        throw new RangeError(`${name} is too large`);
    }
    return result;
}

export class FFTE {
    constructor(module) {
        this.module = module;
    }

    _call(input, outputLength, transform) {
        const inputBytes = input.length * Float64Array.BYTES_PER_ELEMENT;
        const outputBytes = outputLength * Float64Array.BYTES_PER_ELEMENT;
        const inputPointer = this.module._malloc(inputBytes);
        const outputPointer = this.module._malloc(outputBytes);

        if (inputPointer === 0 || outputPointer === 0) {
            if (inputPointer !== 0) this.module._free(inputPointer);
            if (outputPointer !== 0) this.module._free(outputPointer);
            throw new Error('Unable to allocate WebAssembly memory');
        }

        try {
            this.module.HEAPF64.set(input, inputPointer / 8);
            const status = transform(inputPointer, outputPointer);
            if (status !== SUCCESS) {
                throw new Error(`FFTE transform failed with status ${status}`);
            }
            return this.module.HEAPF64
                .slice(outputPointer / 8, outputPointer / 8 + outputLength);
        } finally {
            this.module._free(inputPointer);
            this.module._free(outputPointer);
        }
    }

    r2c1d(values) {
        const input = Float64Array.from(values);
        requirePositiveInteger(input.length, 'input length');
        const complexLength = Math.floor(input.length / 2) + 1;
        return this._call(input, complexLength * 2, (inputPointer, outputPointer) =>
            this.module._ffte_r2c_1d(inputPointer, input.length, outputPointer)
        );
    }

    r2c1dBatch(values, length) {
        requirePositiveInteger(length, 'length');
        const input =
            values instanceof Float64Array ? values : Float64Array.from(values);
        if (input.length === 0 || input.length % length !== 0) {
            throw new RangeError('input length must be a positive multiple of length');
        }
        const batchCount = input.length / length;
        const outputLength =
            batchCount * 2 * (Math.floor(length / 2) + 1);
        return this._call(input, outputLength, (inputPointer, outputPointer) =>
            this.module._ffte_r2c_1d_batch(
                inputPointer,
                length,
                batchCount,
                outputPointer
            )
        );
    }

    c2r1d(spectrum, length) {
        requirePositiveInteger(length, 'length');
        const input = Float64Array.from(spectrum);
        const expectedLength = 2 * (Math.floor(length / 2) + 1);
        if (input.length !== expectedLength) {
            throw new RangeError(`spectrum must contain ${expectedLength} doubles`);
        }
        return this._call(input, length, (inputPointer, outputPointer) =>
            this.module._ffte_c2r_1d(inputPointer, length, outputPointer)
        );
    }

    c2c1d(values, inverse = false) {
        const input = Float64Array.from(values);
        if (input.length === 0 || input.length % 2 !== 0) {
            throw new RangeError('values must contain interleaved complex pairs');
        }
        const length = input.length / 2;
        return this._call(input, input.length, (inputPointer, outputPointer) =>
            this.module._ffte_c2c_1d(
                inputPointer,
                length,
                inverse ? 1 : -1,
                outputPointer
            )
        );
    }

    r2c2d(values, rows, columns) {
        requirePositiveInteger(rows, 'rows');
        requirePositiveInteger(columns, 'columns');
        const input = Float64Array.from(values);
        const expectedLength = checkedProduct(rows, columns, 'input shape');
        if (input.length !== expectedLength) {
            throw new RangeError(`input must contain ${expectedLength} doubles`);
        }
        const outputLength = checkedProduct(
            rows,
            2 * (Math.floor(columns / 2) + 1),
            'output shape'
        );
        return this._call(input, outputLength, (inputPointer, outputPointer) =>
            this.module._ffte_r2c_2d(inputPointer, rows, columns, outputPointer)
        );
    }

    c2r2d(spectrum, rows, columns) {
        requirePositiveInteger(rows, 'rows');
        requirePositiveInteger(columns, 'columns');
        const input = Float64Array.from(spectrum);
        const expectedLength = checkedProduct(
            rows,
            2 * (Math.floor(columns / 2) + 1),
            'spectrum shape'
        );
        if (input.length !== expectedLength) {
            throw new RangeError(`spectrum must contain ${expectedLength} doubles`);
        }
        const outputLength = checkedProduct(rows, columns, 'output shape');
        return this._call(input, outputLength, (inputPointer, outputPointer) =>
            this.module._ffte_c2r_2d(
                inputPointer,
                rows,
                columns,
                outputPointer
            )
        );
    }

    c2c2d(values, rows, columns, inverse = false) {
        requirePositiveInteger(rows, 'rows');
        requirePositiveInteger(columns, 'columns');
        const input = Float64Array.from(values);
        const expectedLength = 2 * checkedProduct(rows, columns, 'input shape');
        if (input.length !== expectedLength) {
            throw new RangeError(`input must contain ${expectedLength} doubles`);
        }
        return this._call(input, input.length, (inputPointer, outputPointer) =>
            this.module._ffte_c2c_2d(
                inputPointer,
                rows,
                columns,
                inverse ? 1 : -1,
                outputPointer
            )
        );
    }
}

export async function createFFT(options = {}) {
    return new FFTE(await createFFTE(options));
}
