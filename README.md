# FFTE

FFTE is a dependency-free C++17 FFT library designed for WebAssembly. It
provides normalized real-to-complex and complex-to-real transforms for
row-major 1D and 2D data of **any positive length**.

- Iterative radix-2 Cooley-Tukey FFT for power-of-two lengths.
- Bluestein's chirp-z algorithm for all other lengths, including primes.
- Compact Hermitian half-spectrum output using interleaved `double` values.
- A stable C ABI plus a small JavaScript `Float64Array` wrapper.

## Spectrum layout

`r2c1d` stores `floor(length / 2) + 1` complex numbers as
`[real0, imag0, real1, imag1, ...]`.

`r2c2d` accepts `rows * columns` row-major real values and stores a row-major
`rows * (floor(columns / 2) + 1)` complex half-spectrum in the same interleaved
format. The inverse transforms accept those layouts and divide by the total
number of samples, so a forward/inverse round trip reproduces the input.

## Native build and test

```sh
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure
```

The public C API is declared in `include/ffte/ffte.h`. All transform functions
return a value from `enum ffte_status`; input and output buffers are owned by
the caller.

## WebAssembly build

Load the Emscripten environment, then configure with its CMake wrapper:

```powershell
& C:\Users\qzhong\emsdk\emsdk_env.ps1
emcmake cmake -S . -B build-wasm -DFFTE_BUILD_TESTS=OFF
cmake --build build-wasm --target ffte_wasm
New-Item -ItemType Directory -Force dist
Copy-Item build-wasm\ffte.js, build-wasm\ffte.wasm dist\
node tests\test_wasm.mjs
```

The generated module uses expandable memory and can run in browsers, web
workers, or Node.js. After copying the two generated files into `dist`, the
JavaScript wrapper can be used as follows:

```js
import { createFFT } from './js/ffte.mjs';

const fft = await createFFT();
const input = new Float64Array([1, 2, 3, 4, 5]);
const spectrum = fft.r2c1d(input);
const restored = fft.c2r1d(spectrum, input.length);

const image = new Float64Array(7 * 11);
const spectrum2d = fft.r2c2d(image, 7, 11);
const restored2d = fft.c2r2d(spectrum2d, 7, 11);
```

Generated `build*` and `dist` directories are intentionally excluded from
version control.
