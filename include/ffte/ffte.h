#ifndef FFTE_FFTE_H
#define FFTE_FFTE_H

#include <stddef.h>

#if defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#define FFTE_API EMSCRIPTEN_KEEPALIVE
#elif defined(_WIN32) && defined(FFTE_SHARED)
#if defined(FFTE_BUILDING_LIBRARY)
#define FFTE_API __declspec(dllexport)
#else
#define FFTE_API __declspec(dllimport)
#endif
#else
#define FFTE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum ffte_status {
    FFTE_SUCCESS = 0,
    FFTE_INVALID_ARGUMENT = 1,
    FFTE_ALLOCATION_FAILURE = 2,
    FFTE_INTERNAL_ERROR = 3
};

enum ffte_direction {
    FFTE_FORWARD = -1,
    FFTE_INVERSE = 1
};

/* Number of complex values produced by a one-dimensional real transform. */
FFTE_API size_t ffte_r2c_1d_complex_size(size_t length);

/* Number of complex values produced by a two-dimensional real transform. */
FFTE_API size_t ffte_r2c_2d_complex_size(size_t rows, size_t columns);

/*
 * Forward 1D transform.
 *
 * input:  length real doubles.
 * output: 2 * ffte_r2c_1d_complex_size(length) doubles containing
 *         interleaved real/imaginary pairs.
 */
FFTE_API int ffte_r2c_1d(const double* input, size_t length, double* output);

/*
 * Forward batch of contiguous, equal-length 1D real transforms.
 *
 * input:  batch_count * length real doubles, transform-major.
 * output: batch_count contiguous interleaved Hermitian half-spectra.
 */
FFTE_API int ffte_r2c_1d_batch(
    const double* input,
    size_t length,
    size_t batch_count,
    double* output
);

/*
 * Inverse 1D transform. The inverse is normalized, so c2r(r2c(x)) == x.
 *
 * input:  interleaved Hermitian half-spectrum containing
 *         ffte_r2c_1d_complex_size(length) complex values.
 * output: length real doubles.
 */
FFTE_API int ffte_c2r_1d(const double* input, size_t length, double* output);

/*
 * Forward or normalized-inverse 1D complex transform.
 * Both buffers contain 2 * length interleaved real/imaginary doubles.
 */
FFTE_API int ffte_c2c_1d(
    const double* input,
    size_t length,
    int direction,
    double* output
);

/*
 * Batch of contiguous, equal-length 1D complex transforms.
 * Both buffers contain 2 * batch_count * length interleaved doubles.
 */
FFTE_API int ffte_c2c_1d_batch(
    const double* input,
    size_t length,
    size_t batch_count,
    int direction,
    double* output
);

/*
 * Forward row-major 2D transform.
 *
 * input:  rows * columns real doubles.
 * output: rows * (columns / 2 + 1) interleaved complex values.
 */
FFTE_API int ffte_r2c_2d(
    const double* input,
    size_t rows,
    size_t columns,
    double* output
);

/*
 * Inverse row-major 2D transform. The inverse is normalized.
 *
 * input:  rows * (columns / 2 + 1) interleaved complex values.
 * output: rows * columns real doubles.
 */
FFTE_API int ffte_c2r_2d(
    const double* input,
    size_t rows,
    size_t columns,
    double* output
);

/*
 * Forward or normalized-inverse row-major 2D complex transform.
 * Both buffers contain 2 * rows * columns interleaved doubles.
 */
FFTE_API int ffte_c2c_2d(
    const double* input,
    size_t rows,
    size_t columns,
    int direction,
    double* output
);

#ifdef __cplusplus
}
#endif

#endif
