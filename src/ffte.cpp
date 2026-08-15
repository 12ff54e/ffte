#include "ffte/ffte.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

namespace {

using Complex = std::complex<double>;

constexpr double kPi = 3.141592653589793238462643383279502884;

bool is_power_of_two(std::size_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

bool product_fits(std::size_t left, std::size_t right) {
    return right == 0 || left <= std::numeric_limits<std::size_t>::max() / right;
}

std::size_t convolution_size(std::size_t length) {
    if (length > (std::numeric_limits<std::size_t>::max() / 2) + 1) {
        throw std::length_error("FFT length is too large");
    }

    const std::size_t required = length * 2 - 1;
    std::size_t result = 1;
    while (result < required) {
        if (result > std::numeric_limits<std::size_t>::max() / 2) {
            throw std::length_error("FFT length is too large");
        }
        result *= 2;
    }
    return result;
}

void radix2_fft(std::vector<Complex>& values, bool inverse) {
    const std::size_t length = values.size();
    if (length <= 1) {
        return;
    }

    for (std::size_t index = 1, reversed = 0; index < length; ++index) {
        std::size_t bit = length >> 1;
        while ((reversed & bit) != 0) {
            reversed ^= bit;
            bit >>= 1;
        }
        reversed ^= bit;
        if (index < reversed) {
            std::swap(values[index], values[reversed]);
        }
    }

    for (std::size_t span = 2;; span *= 2) {
        const double angle = (inverse ? 2.0 : -2.0) * kPi /
                             static_cast<double>(span);
        const Complex step(std::cos(angle), std::sin(angle));

        for (std::size_t offset = 0; offset < length; offset += span) {
            Complex twiddle(1.0, 0.0);
            const std::size_t half = span / 2;
            for (std::size_t index = 0; index < half; ++index) {
                const Complex even = values[offset + index];
                const Complex odd = values[offset + index + half] * twiddle;
                values[offset + index] = even + odd;
                values[offset + index + half] = even - odd;
                twiddle *= step;
            }
        }

        if (span == length) {
            break;
        }
    }

    if (inverse) {
        const double scale = 1.0 / static_cast<double>(length);
        for (Complex& value : values) {
            value *= scale;
        }
    }
}

void bluestein_fft(std::vector<Complex>& values, bool inverse) {
    const std::size_t length = values.size();
    const std::size_t padded_length = convolution_size(length);
    std::vector<Complex> left(padded_length, Complex(0.0, 0.0));
    std::vector<Complex> right(padded_length, Complex(0.0, 0.0));

    for (std::size_t index = 0; index < length; ++index) {
        const long double position = static_cast<long double>(index);
        const double angle = static_cast<double>(
            static_cast<long double>(kPi) * position * position /
            static_cast<long double>(length)
        );
        const double signed_angle = inverse ? angle : -angle;
        const Complex input_chirp(std::cos(signed_angle), std::sin(signed_angle));
        const Complex convolution_chirp(
            std::cos(-signed_angle),
            std::sin(-signed_angle)
        );

        left[index] = values[index] * input_chirp;
        right[index] = convolution_chirp;
        if (index != 0) {
            right[padded_length - index] = convolution_chirp;
        }
    }

    radix2_fft(left, false);
    radix2_fft(right, false);
    for (std::size_t index = 0; index < padded_length; ++index) {
        left[index] *= right[index];
    }
    radix2_fft(left, true);

    const double scale = inverse ? 1.0 / static_cast<double>(length) : 1.0;
    for (std::size_t index = 0; index < length; ++index) {
        const long double position = static_cast<long double>(index);
        const double angle = static_cast<double>(
            static_cast<long double>(kPi) * position * position /
            static_cast<long double>(length)
        );
        const double signed_angle = inverse ? angle : -angle;
        const Complex output_chirp(std::cos(signed_angle), std::sin(signed_angle));
        values[index] = left[index] * output_chirp * scale;
    }
}

void complex_fft(std::vector<Complex>& values, bool inverse) {
    if (values.size() <= 1) {
        return;
    }
    if (is_power_of_two(values.size())) {
        radix2_fft(values, inverse);
    } else {
        bluestein_fft(values, inverse);
    }
}

void complex_fft_2d(
    std::vector<Complex>& values,
    std::size_t rows,
    std::size_t columns,
    bool inverse
) {
    std::vector<Complex> line(std::max(rows, columns));

    for (std::size_t row = 0; row < rows; ++row) {
        line.resize(columns);
        std::copy_n(values.begin() + row * columns, columns, line.begin());
        complex_fft(line, inverse);
        std::copy(line.begin(), line.end(), values.begin() + row * columns);
    }

    for (std::size_t column = 0; column < columns; ++column) {
        line.resize(rows);
        for (std::size_t row = 0; row < rows; ++row) {
            line[row] = values[row * columns + column];
        }
        complex_fft(line, inverse);
        for (std::size_t row = 0; row < rows; ++row) {
            values[row * columns + column] = line[row];
        }
    }
}

template <typename Operation>
int safely(Operation&& operation) {
    try {
        operation();
        return FFTE_SUCCESS;
    } catch (const std::bad_alloc&) {
        return FFTE_ALLOCATION_FAILURE;
    } catch (...) {
        return FFTE_INTERNAL_ERROR;
    }
}

}  // namespace

extern "C" {

std::size_t ffte_r2c_1d_complex_size(std::size_t length) {
    return length == 0 ? 0 : length / 2 + 1;
}

std::size_t ffte_r2c_2d_complex_size(std::size_t rows, std::size_t columns) {
    if (rows == 0 || columns == 0) {
        return 0;
    }
    const std::size_t half_columns = columns / 2 + 1;
    return product_fits(rows, half_columns) ? rows * half_columns : 0;
}

int ffte_r2c_1d(const double* input, std::size_t length, double* output) {
    if (input == nullptr || output == nullptr || length == 0) {
        return FFTE_INVALID_ARGUMENT;
    }

    return safely([&] {
        std::vector<Complex> spectrum(length);
        for (std::size_t index = 0; index < length; ++index) {
            spectrum[index] = Complex(input[index], 0.0);
        }
        complex_fft(spectrum, false);

        const std::size_t output_length = ffte_r2c_1d_complex_size(length);
        for (std::size_t index = 0; index < output_length; ++index) {
            output[index * 2] = spectrum[index].real();
            output[index * 2 + 1] = spectrum[index].imag();
        }
    });
}

int ffte_c2r_1d(const double* input, std::size_t length, double* output) {
    if (input == nullptr || output == nullptr || length == 0) {
        return FFTE_INVALID_ARGUMENT;
    }

    return safely([&] {
        std::vector<Complex> spectrum(length, Complex(0.0, 0.0));
        const std::size_t input_length = ffte_r2c_1d_complex_size(length);
        for (std::size_t index = 0; index < input_length; ++index) {
            spectrum[index] = Complex(input[index * 2], input[index * 2 + 1]);
        }
        for (std::size_t index = input_length; index < length; ++index) {
            spectrum[index] = std::conj(spectrum[length - index]);
        }

        complex_fft(spectrum, true);
        for (std::size_t index = 0; index < length; ++index) {
            output[index] = spectrum[index].real();
        }
    });
}

int ffte_c2c_1d(
    const double* input,
    std::size_t length,
    int direction,
    double* output
) {
    if (input == nullptr || output == nullptr || length == 0 ||
        (direction != FFTE_FORWARD && direction != FFTE_INVERSE)) {
        return FFTE_INVALID_ARGUMENT;
    }

    return safely([&] {
        std::vector<Complex> values(length);
        for (std::size_t index = 0; index < length; ++index) {
            values[index] = Complex(input[index * 2], input[index * 2 + 1]);
        }
        complex_fft(values, direction == FFTE_INVERSE);
        for (std::size_t index = 0; index < length; ++index) {
            output[index * 2] = values[index].real();
            output[index * 2 + 1] = values[index].imag();
        }
    });
}

int ffte_r2c_2d(
    const double* input,
    std::size_t rows,
    std::size_t columns,
    double* output
) {
    if (input == nullptr || output == nullptr || rows == 0 || columns == 0 ||
        !product_fits(rows, columns)) {
        return FFTE_INVALID_ARGUMENT;
    }

    return safely([&] {
        const std::size_t value_count = rows * columns;
        std::vector<Complex> spectrum(value_count);
        for (std::size_t index = 0; index < value_count; ++index) {
            spectrum[index] = Complex(input[index], 0.0);
        }
        complex_fft_2d(spectrum, rows, columns, false);

        const std::size_t half_columns = columns / 2 + 1;
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < half_columns; ++column) {
                const Complex value = spectrum[row * columns + column];
                const std::size_t output_index = (row * half_columns + column) * 2;
                output[output_index] = value.real();
                output[output_index + 1] = value.imag();
            }
        }
    });
}

int ffte_c2r_2d(
    const double* input,
    std::size_t rows,
    std::size_t columns,
    double* output
) {
    if (input == nullptr || output == nullptr || rows == 0 || columns == 0 ||
        !product_fits(rows, columns)) {
        return FFTE_INVALID_ARGUMENT;
    }

    return safely([&] {
        const std::size_t half_columns = columns / 2 + 1;
        std::vector<Complex> spectrum(rows * columns, Complex(0.0, 0.0));

        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < half_columns; ++column) {
                const std::size_t input_index = (row * half_columns + column) * 2;
                spectrum[row * columns + column] =
                    Complex(input[input_index], input[input_index + 1]);
            }
        }

        for (std::size_t row = 0; row < rows; ++row) {
            const std::size_t mirror_row = (rows - row) % rows;
            for (std::size_t column = half_columns; column < columns; ++column) {
                const std::size_t mirror_column = columns - column;
                spectrum[row * columns + column] =
                    std::conj(spectrum[mirror_row * columns + mirror_column]);
            }
        }

        complex_fft_2d(spectrum, rows, columns, true);
        for (std::size_t index = 0; index < rows * columns; ++index) {
            output[index] = spectrum[index].real();
        }
    });
}

int ffte_c2c_2d(
    const double* input,
    std::size_t rows,
    std::size_t columns,
    int direction,
    double* output
) {
    if (input == nullptr || output == nullptr || rows == 0 || columns == 0 ||
        !product_fits(rows, columns) ||
        (direction != FFTE_FORWARD && direction != FFTE_INVERSE)) {
        return FFTE_INVALID_ARGUMENT;
    }

    return safely([&] {
        const std::size_t value_count = rows * columns;
        std::vector<Complex> values(value_count);
        for (std::size_t index = 0; index < value_count; ++index) {
            values[index] = Complex(input[index * 2], input[index * 2 + 1]);
        }
        complex_fft_2d(values, rows, columns, direction == FFTE_INVERSE);
        for (std::size_t index = 0; index < value_count; ++index) {
            output[index * 2] = values[index].real();
            output[index * 2 + 1] = values[index].imag();
        }
    });
}

}  // extern "C"
