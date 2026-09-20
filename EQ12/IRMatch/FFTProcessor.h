
/*
 * FFTProcessor.h
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Copyright (C) 2026 brummer <brummer@web.de>
 */

#pragma once

#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>

#ifdef PFFFT_SUPPORT
#include <limits>
#include <stdexcept>
#include "../pffft/pffft.h"
#else
#include <fftw3.h>
#endif

/****************************************************************
 * @file FFTProcessor.h
 * @brief FFTW-based forward/inverse FFT helpers,
 * plus spectral division and minimum-phase (cepstral) reconstruction.
 ****************************************************************/

class FFTProcessor {
public:
    using Complex = std::complex<double>;
    using CVec = std::vector<Complex>;

    static CVec fft(const CVec& in) {
#ifdef PFFFT_SUPPORT
        return transform(in, PFFFT_FORWARD, false);
#else
        int N = (int)in.size();

        fftw_complex *input = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
        fftw_complex *output = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);

        for (int i = 0; i < N; ++i) {
            input[i][0] = in[i].real();
            input[i][1] = in[i].imag();
        }

        fftw_plan p = fftw_plan_dft_1d(N, input, output, FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_execute(p);

        CVec out(N);
        for (int i = 0; i < N; ++i)
            out[i] = Complex(output[i][0], output[i][1]);

        fftw_destroy_plan(p);
        fftw_free(input);
        fftw_free(output);

        return out;
#endif
    }

    static CVec ifft(const CVec& in) {
#ifdef PFFFT_SUPPORT
        return transform(in, PFFFT_BACKWARD, true);
#else
        int N = (int)in.size();

        fftw_complex *input = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
        fftw_complex *output = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);

        for (int i = 0; i < N; ++i) {
            input[i][0] = in[i].real();
            input[i][1] = in[i].imag();
        }

        fftw_plan p = fftw_plan_dft_1d(N, input, output, FFTW_BACKWARD, FFTW_ESTIMATE);
        fftw_execute(p);

        CVec out(N);
        for (int i = 0; i < N; ++i)
            out[i] = Complex(output[i][0] / N, output[i][1] / N);

        fftw_destroy_plan(p);
        fftw_free(input);
        fftw_free(output);

        return out;
#endif
    }

    static CVec safe_divide(const CVec& a, const CVec& b) {
        size_t n = a.size();
        CVec out(n);

        for (size_t i = 0; i < n; ++i) {
            double denom = std::norm(b[i]) + EPS;
            out[i] = a[i] * std::conj(b[i]) / denom;
        }

        return out;
    }

    static CVec mps(const CVec& s) {
        CVec log_s(s.size());

        for (size_t i = 0; i < s.size(); ++i)
            log_s[i] = std::log(std::max<double>(std::abs(s[i]), EPS));

        CVec cp = ifft(log_s);
        fold(cp);
        CVec out = fft(cp);

        for (auto& v : out)
            v = std::exp(v);

        return out;
    }

private:
    static constexpr double EPS = 1e-12;

#ifdef PFFFT_SUPPORT
    static CVec transform(const CVec& in,
                          pffft_direction_t direction,
                          bool normalize) {
        const size_t n = in.size();

        if (n == 0)
            return {};

        if (n > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("FFT size is too large for PFFFT");

        const int N = static_cast<int>(n);

        PFFFT_Setup* setup =
            pffft_new_setup(N, PFFFT_COMPLEX);

        if (!setup)
            throw std::invalid_argument(
                "FFT size is not supported by PFFFT");

        // PFFFT complex buffers contain N interleaved complex values,
        // therefore each buffer has 2*N floats.
        float* input = static_cast<float*>(
            pffft_aligned_malloc(sizeof(float) * 2 * N));
        float* output = static_cast<float*>(
            pffft_aligned_malloc(sizeof(float) * 2 * N));

        if (!input || !output) {
            pffft_aligned_free(input);
            pffft_aligned_free(output);
            pffft_destroy_setup(setup);
            throw std::bad_alloc();
        }

        for (int i = 0; i < N; ++i) {
            input[2 * i + 0] = static_cast<float>(in[i].real());
            input[2 * i + 1] = static_cast<float>(in[i].imag());
        }

        pffft_transform_ordered(
            setup,
            input,
            output,
            nullptr,
            direction);

        CVec out(n);
        const double scale = normalize
            ? 1.0 / static_cast<double>(N)
            : 1.0;

        for (int i = 0; i < N; ++i) {
            out[i] = Complex(
                static_cast<double>(output[2 * i + 0]) * scale,
                static_cast<double>(output[2 * i + 1]) * scale);
        }

        pffft_aligned_free(input);
        pffft_aligned_free(output);
        pffft_destroy_setup(setup);

        return out;
    }
#endif

    // Cepstrum min-phase
    static void fold(CVec& r) {
        size_t n = r.size();
        size_t nt = n / 2;

        for (size_t i = 1; i < nt; ++i)
            r[i] += std::conj(r[n - i]);

        for (size_t i = nt + 1; i < n; ++i)
            r[i] = 0.0;
    }

};
