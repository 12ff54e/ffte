#include "ffte/ffte.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

using Complex = std::complex<double>;

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kTolerance = 2e-9;
int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

double sample(std::size_t index) {
    const double x = static_cast<double>(index);
    return std::sin(x * 0.37) + 0.2 * std::cos(x * 1.13) + (index % 5) * 0.03;
}

std::vector<Complex> reference_dft_1d(const std::vector<double>& input) {
    std::vector<Complex> output(input.size());
    for (std::size_t frequency = 0; frequency < input.size(); ++frequency) {
        for (std::size_t position = 0; position < input.size(); ++position) {
            const double angle = -2.0 * kPi * static_cast<double>(frequency) *
                                 static_cast<double>(position) /
                                 static_cast<double>(input.size());
            output[frequency] += input[position] *
                                 Complex(std::cos(angle), std::sin(angle));
        }
    }
    return output;
}

std::vector<Complex> reference_complex_dft_1d(
    const std::vector<Complex>& input
) {
    std::vector<Complex> output(input.size());
    for (std::size_t frequency = 0; frequency < input.size(); ++frequency) {
        for (std::size_t position = 0; position < input.size(); ++position) {
            const double angle = -2.0 * kPi * static_cast<double>(frequency) *
                                 static_cast<double>(position) /
                                 static_cast<double>(input.size());
            output[frequency] += input[position] *
                                 Complex(std::cos(angle), std::sin(angle));
        }
    }
    return output;
}

std::vector<Complex> reference_dft_2d(
    const std::vector<double>& input,
    std::size_t rows,
    std::size_t columns
) {
    std::vector<Complex> output(rows * columns);
    for (std::size_t output_row = 0; output_row < rows; ++output_row) {
        for (std::size_t output_column = 0; output_column < columns;
             ++output_column) {
            Complex sum(0.0, 0.0);
            for (std::size_t row = 0; row < rows; ++row) {
                for (std::size_t column = 0; column < columns; ++column) {
                    const double angle = -2.0 * kPi *
                        (static_cast<double>(output_row * row) /
                             static_cast<double>(rows) +
                         static_cast<double>(output_column * column) /
                             static_cast<double>(columns));
                    sum += input[row * columns + column] *
                           Complex(std::cos(angle), std::sin(angle));
                }
            }
            output[output_row * columns + output_column] = sum;
        }
    }
    return output;
}

void test_1d(std::size_t length) {
    std::vector<double> input(length);
    for (std::size_t index = 0; index < length; ++index) {
        input[index] = sample(index);
    }

    const std::size_t complex_length = ffte_r2c_1d_complex_size(length);
    std::vector<double> spectrum(complex_length * 2);
    check(
        ffte_r2c_1d(input.data(), length, spectrum.data()) == FFTE_SUCCESS,
        "1D forward status for length " + std::to_string(length)
    );

    if (length <= 17) {
        const auto expected = reference_dft_1d(input);
        for (std::size_t index = 0; index < complex_length; ++index) {
            const Complex actual(spectrum[index * 2], spectrum[index * 2 + 1]);
            check(
                std::abs(actual - expected[index]) < kTolerance,
                "1D reference result for length " + std::to_string(length) +
                    ", bin " + std::to_string(index)
            );
        }
    }

    std::vector<double> reconstructed(length);
    check(
        ffte_c2r_1d(spectrum.data(), length, reconstructed.data()) == FFTE_SUCCESS,
        "1D inverse status for length " + std::to_string(length)
    );
    for (std::size_t index = 0; index < length; ++index) {
        check(
            std::abs(reconstructed[index] - input[index]) < kTolerance,
            "1D round trip for length " + std::to_string(length) +
                ", sample " + std::to_string(index)
        );
    }
}

void test_real_batch(std::size_t length, std::size_t batch_count) {
    const std::size_t complex_length = ffte_r2c_1d_complex_size(length);
    const std::size_t output_stride = complex_length * 2;
    std::vector<double> input(length * batch_count);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = sample(index) + 0.01 * static_cast<double>(index / length);
    }
    std::vector<double> output(batch_count * output_stride);
    check(
        ffte_r2c_1d_batch(
            input.data(), length, batch_count, output.data()
        ) == FFTE_SUCCESS,
        "batched 1D forward status for length " + std::to_string(length)
    );

    for (std::size_t batch = 0; batch < batch_count; ++batch) {
        const std::vector<double> transform_input(
            input.begin() + batch * length,
            input.begin() + (batch + 1) * length
        );
        const auto expected = reference_dft_1d(transform_input);
        for (std::size_t index = 0; index < complex_length; ++index) {
            const std::size_t output_index = batch * output_stride + index * 2;
            const Complex actual(output[output_index], output[output_index + 1]);
            check(
                std::abs(actual - expected[index]) < kTolerance,
                "batched 1D result for length " + std::to_string(length) +
                    ", batch " + std::to_string(batch) + ", bin " +
                    std::to_string(index)
            );
        }
    }
}

void test_complex_1d(std::size_t length) {
    std::vector<Complex> input(length);
    std::vector<double> interleaved(length * 2);
    for (std::size_t index = 0; index < length; ++index) {
        input[index] = Complex(sample(index), sample(index + 3) * 0.4);
        interleaved[index * 2] = input[index].real();
        interleaved[index * 2 + 1] = input[index].imag();
    }

    std::vector<double> spectrum(length * 2);
    check(
        ffte_c2c_1d(
            interleaved.data(), length, FFTE_FORWARD, spectrum.data()
        ) == FFTE_SUCCESS,
        "complex 1D forward status for length " + std::to_string(length)
    );

    if (length <= 17) {
        const auto expected = reference_complex_dft_1d(input);
        for (std::size_t index = 0; index < length; ++index) {
            const Complex actual(spectrum[index * 2], spectrum[index * 2 + 1]);
            check(
                std::abs(actual - expected[index]) < kTolerance,
                "complex 1D reference result for length " +
                    std::to_string(length) + ", bin " + std::to_string(index)
            );
        }
    }

    std::vector<double> reconstructed(length * 2);
    check(
        ffte_c2c_1d(
            spectrum.data(), length, FFTE_INVERSE, reconstructed.data()
        ) == FFTE_SUCCESS,
        "complex 1D inverse status for length " + std::to_string(length)
    );
    for (std::size_t index = 0; index < interleaved.size(); ++index) {
        check(
            std::abs(reconstructed[index] - interleaved[index]) < kTolerance,
            "complex 1D round trip for length " + std::to_string(length) +
                ", component " + std::to_string(index)
        );
    }
}

void test_complex_batch(std::size_t length, std::size_t batch_count) {
    const std::size_t stride = length * 2;
    std::vector<double> input(batch_count * stride);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = sample(index) * 0.7 + index * 0.0003;
    }
    std::vector<double> spectrum(input.size());
    std::vector<double> reconstructed(input.size());

    check(
        ffte_c2c_1d_batch(
            input.data(), length, batch_count, FFTE_FORWARD, spectrum.data()
        ) == FFTE_SUCCESS,
        "complex batch forward status for length " + std::to_string(length)
    );
    for (std::size_t batch = 0; batch < batch_count; ++batch) {
        std::vector<Complex> transform_input(length);
        for (std::size_t index = 0; index < length; ++index) {
            const std::size_t offset = batch * stride + index * 2;
            transform_input[index] = Complex(input[offset], input[offset + 1]);
        }
        const auto expected = reference_complex_dft_1d(transform_input);
        for (std::size_t index = 0; index < length; ++index) {
            const std::size_t offset = batch * stride + index * 2;
            const Complex actual(spectrum[offset], spectrum[offset + 1]);
            check(
                std::abs(actual - expected[index]) < kTolerance,
                "complex batch result for length " + std::to_string(length) +
                    ", batch " + std::to_string(batch) + ", bin " +
                    std::to_string(index)
            );
        }
    }

    check(
        ffte_c2c_1d_batch(
            spectrum.data(),
            length,
            batch_count,
            FFTE_INVERSE,
            reconstructed.data()
        ) == FFTE_SUCCESS,
        "complex batch inverse status for length " + std::to_string(length)
    );
    for (std::size_t index = 0; index < input.size(); ++index) {
        check(
            std::abs(reconstructed[index] - input[index]) < kTolerance,
            "complex batch round trip for component " + std::to_string(index)
        );
    }
}

void test_2d(std::size_t rows, std::size_t columns) {
    std::vector<double> input(rows * columns);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = sample(index) + 0.01 * static_cast<double>(index / columns);
    }

    const std::size_t half_columns = columns / 2 + 1;
    const std::size_t complex_length = ffte_r2c_2d_complex_size(rows, columns);
    std::vector<double> spectrum(complex_length * 2);
    check(
        ffte_r2c_2d(input.data(), rows, columns, spectrum.data()) == FFTE_SUCCESS,
        "2D forward status for " + std::to_string(rows) + "x" +
            std::to_string(columns)
    );

    if (rows * columns <= 36) {
        const auto expected = reference_dft_2d(input, rows, columns);
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < half_columns; ++column) {
                const std::size_t output_index = row * half_columns + column;
                const Complex actual(
                    spectrum[output_index * 2],
                    spectrum[output_index * 2 + 1]
                );
                check(
                    std::abs(actual - expected[row * columns + column]) <
                        kTolerance,
                    "2D reference result for " + std::to_string(rows) + "x" +
                        std::to_string(columns) + ", bin " +
                        std::to_string(row) + "," + std::to_string(column)
                );
            }
        }
    }

    std::vector<double> reconstructed(rows * columns);
    check(
        ffte_c2r_2d(
            spectrum.data(), rows, columns, reconstructed.data()
        ) == FFTE_SUCCESS,
        "2D inverse status for " + std::to_string(rows) + "x" +
            std::to_string(columns)
    );
    for (std::size_t index = 0; index < input.size(); ++index) {
        check(
            std::abs(reconstructed[index] - input[index]) < kTolerance,
            "2D round trip for " + std::to_string(rows) + "x" +
                std::to_string(columns) + ", sample " + std::to_string(index)
        );
    }
}

void test_complex_2d(std::size_t rows, std::size_t columns) {
    const std::size_t value_count = rows * columns;
    std::vector<double> input(value_count * 2);
    for (std::size_t index = 0; index < value_count; ++index) {
        input[index * 2] = sample(index);
        input[index * 2 + 1] = sample(index + 5) * 0.3;
    }

    std::vector<double> spectrum(input.size());
    std::vector<double> reconstructed(input.size());
    check(
        ffte_c2c_2d(
            input.data(), rows, columns, FFTE_FORWARD, spectrum.data()
        ) == FFTE_SUCCESS,
        "complex 2D forward status for " + std::to_string(rows) + "x" +
            std::to_string(columns)
    );
    check(
        ffte_c2c_2d(
            spectrum.data(), rows, columns, FFTE_INVERSE, reconstructed.data()
        ) == FFTE_SUCCESS,
        "complex 2D inverse status for " + std::to_string(rows) + "x" +
            std::to_string(columns)
    );
    for (std::size_t index = 0; index < input.size(); ++index) {
        check(
            std::abs(reconstructed[index] - input[index]) < kTolerance,
            "complex 2D round trip for " + std::to_string(rows) + "x" +
                std::to_string(columns) + ", component " +
                std::to_string(index)
        );
    }
}

void test_invalid_arguments() {
    double value = 0.0;
    check(ffte_r2c_1d(nullptr, 1, &value) == FFTE_INVALID_ARGUMENT,
          "1D forward rejects null input");
    check(ffte_r2c_1d(&value, 0, &value) == FFTE_INVALID_ARGUMENT,
          "1D forward rejects zero length");
    check(ffte_r2c_1d_batch(&value, 1, 0, &value) == FFTE_INVALID_ARGUMENT,
          "batched 1D forward rejects zero batches");
    check(ffte_c2r_1d(&value, 1, nullptr) == FFTE_INVALID_ARGUMENT,
          "1D inverse rejects null output");
    check(ffte_r2c_2d(&value, 0, 1, &value) == FFTE_INVALID_ARGUMENT,
          "2D forward rejects zero rows");
    check(ffte_c2r_2d(&value, 1, 0, &value) == FFTE_INVALID_ARGUMENT,
          "2D inverse rejects zero columns");
    check(ffte_c2c_1d(&value, 1, 0, &value) == FFTE_INVALID_ARGUMENT,
          "complex 1D rejects invalid direction");
    check(ffte_c2c_1d_batch(&value, 1, 0, FFTE_FORWARD, &value) ==
              FFTE_INVALID_ARGUMENT,
          "complex batch rejects zero batches");
    check(ffte_c2c_2d(&value, 1, 1, 0, &value) == FFTE_INVALID_ARGUMENT,
          "complex 2D rejects invalid direction");
}

}  // namespace

int main() {
    for (const std::size_t length : {
             1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U, 10U, 12U, 15U,
             16U, 17U, 25U, 31U, 32U, 63U, 64U, 65U, 97U, 127U
         }) {
        test_1d(length);
        test_real_batch(length, 3);
        test_complex_1d(length);
        test_complex_batch(length, 3);
    }

    for (const auto dimensions : {
             std::pair<std::size_t, std::size_t>{1, 1}, {1, 5}, {5, 1},
             {2, 3}, {3, 2}, {3, 5}, {4, 4}, {5, 7}, {6, 10}, {7, 11}
         }) {
        test_2d(dimensions.first, dimensions.second);
        test_complex_2d(dimensions.first, dimensions.second);
    }

    test_invalid_arguments();

    if (failures != 0) {
        std::cerr << failures << " test assertion(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "All FFTE tests passed\n";
    return EXIT_SUCCESS;
}
