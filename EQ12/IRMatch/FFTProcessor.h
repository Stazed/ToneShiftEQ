
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
#include <map>

#include "AudioFFT.h"

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
        const size_t N = in.size();
        std::vector<double> x(N), re(N/2 + 1), im(N/2 + 1);
        for (size_t i = 0; i < N; ++i) x[i] = in[i].real();
        audiofft::AudioFFT f;
        f.init(N);
        f.fft(x.data(), re.data(), im.data());
        CVec out(N);
        for (size_t k = 0; k <= N/2; ++k) out[k] = Complex(re[k], im[k]);
        for (size_t k = N/2 + 1; k < N; ++k) out[k] = std::conj(out[N - k]);
        return out;
    }

    static CVec ifft(const CVec& in) {
        const size_t N = in.size();
        std::vector<double> re(N/2 + 1), im(N/2 + 1), x(N);
        for (size_t k = 0; k <= N/2; ++k) { re[k] = in[k].real(); im[k] = in[k].imag(); }
        audiofft::AudioFFT f;
        f.init(N);
        f.ifft(x.data(), re.data(), im.data());
        CVec out(N);
        for (size_t i = 0; i < N; ++i) out[i] = Complex(x[i], 0.0);
        return out;
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
