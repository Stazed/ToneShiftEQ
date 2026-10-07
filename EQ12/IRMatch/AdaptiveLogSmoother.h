
/*
 * AdaptiveLogSmoother.h
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Copyright (C) 2026 brummer <brummer@web.de>
 */

#pragma once

#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>

/****************************************************************
 * Adaptive logarithmic spectrum smoother.
 *
 * Gaussian smoothing with constant width in octaves (sigmaOct),
 * truncated at +-3 sigma.
 *
 *  - Low bins (narrow window): direct weighted sum.
 *  - Higher bins: the spectrum is averaged into cells of constant
 *    width in octaves (sigma/6) using a prefix sum, smoothed with ONE
 *    fixed 37-tap Gaussian on that log grid and interpolated back
 *    onto the linear bins. Cost is O(bins), no per-bin kernels.
 *
 * Thread safety: scratch buffers are members; use one instance
 * per thread. `out` must not alias `in`.
****************************************************************/

template<typename Vec>
class AdaptiveLogSmoother {
public:
    void prepare(size_t bins, double sampleRate, double sigmaOct = 0.18) {
        if (bins_ == bins && sr_ == sampleRate && sigma_ == sigmaOct) return;
        bins_ = bins;
        sr_ = sampleRate;
        sigma_ = sigmaOct;
        build();
    }

    void process(const Vec& in, Vec& out) {
        const size_t n = in.size();
        if (out.size() != n) out.resize(n);
        if (n == 0) return;
        if (n != bins_) {
            if (sr_ <= 0.0) {
                out = in;
                return;
            }
            prepare(n, sr_, sigma_);
        }
        out[0] = in[0];
        if (n < 2) return;

        // ---- low bins: direct ----
        for (size_t i = 1; i < directEnd_; ++i) {
            const float* w = dW_.data() + dOff_[i];
            const size_t cnt = dOff_[i + 1] - dOff_[i];
            const double* src = in.data() + dFirst_[i];
            double sum = 0.0;
            for (size_t j = 0; j < cnt; ++j)
                sum += src[j] * (double)w[j];
            out[i] = sum;
        }
        if (directEnd_ >= n) return;

        // ---- higher bins: log grid ----
        S_[0] = 0.0;
        for (size_t i = 0; i < n; ++i)
            S_[i + 1] = S_[i] + in[i];

        for (size_t k = 0; k < K_; ++k) {
            const double a = S_[jLo_[k]] + in[jLo_[k]] * fLo_[k];
            const double b = S_[jHi_[k]] + in[jHi_[k]] * fHi_[k];
            c_[k] = (b - a) * invW_[k];
        }

        const int R = (int)(g_.size() / 2);
        const int Ki = (int)K_;
        for (int k = 0; k < Ki; ++k) {
            const int m0 = std::max(-R, -k);
            const int m1 = std::min(R, Ki - 1 - k);
            double s = 0.0;
            for (int m = m0; m <= m1; ++m)
                s += g_[m + R] * c_[k + m];
            sm_[k] = s * invG_[k];
        }

        for (size_t i = directEnd_; i < n; ++i) {
            const size_t q = i - directEnd_;
            const size_t k = ik_[q];
            const double t = it_[q];
            out[i] = sm_[k] + (sm_[k + 1] - sm_[k]) * t;
        }
    }

    Vec process(const Vec& in) {
        process(in, smoothBuffer_);
        return smoothBuffer_;
    }

    void reset() {
        bins_ = 0;
        sr_ = 0.0;
        sigma_ = 0.0;
    }

private:
    static constexpr double kDirectTaps     = 48.0; // full window size below which bins are done directly
    static constexpr double kCellsPerSigma  = 6.0;
    static constexpr double kRadius         = 3.0;  // in sigma

    size_t bins_ = 0;
    double sr_ = 0.0;
    double sigma_ = 0.0;

    // direct region (bins 1 .. directEnd_-1)
    size_t directEnd_ = 0;
    std::vector<uint32_t> dFirst_;
    std::vector<uint32_t> dOff_;
    std::vector<float>    dW_;

    // log grid region
    size_t K_ = 0;
    std::vector<uint32_t> jLo_;
    std::vector<uint32_t> jHi_;
    std::vector<double>   fLo_;
    std::vector<double>   fHi_;
    std::vector<double>   invW_;
    std::vector<double>   g_;
    std::vector<double>   invG_;
    std::vector<uint32_t> ik_;
    std::vector<float>    it_;
    std::vector<double>   S_;
    std::vector<double>   c_;
    std::vector<double>   sm_;
    mutable Vec smoothBuffer_;

    void build() {
        const size_t n = bins_;
        dFirst_.clear();
        dOff_.clear();
        dW_.clear();
        jLo_.clear();
        jHi_.clear();
        fLo_.clear();
        fHi_.clear();
        invW_.clear();
        g_.clear();
        invG_.clear();
        ik_.clear();
        it_.clear();

        K_ = 0;
        directEnd_ = 0;

        if (n < 2 || sr_ <= 0.0 || sigma_ <= 0.0) return;

        const double r2 = std::exp2(kRadius * sigma_);
        directEnd_ = std::clamp<size_t>((size_t)std::ceil(kDirectTaps / (r2 - 1.0 / r2)), 2, n);

        // direct kernels: log-domain Gaussian, each linear bin weighted by 1/j
        dFirst_.assign(directEnd_, 0);
        dOff_.assign(directEnd_ + 1, 0);
        for (size_t i = 1; i < directEnd_; ++i) {
            size_t j1 = std::max<size_t>(1, (size_t)std::ceil((double)i / r2));
            size_t j2 = std::min<size_t>(n - 1, (size_t)std::floor((double)i * r2));
            dFirst_[i] = (uint32_t)j1;
            dOff_[i]   = (uint32_t)dW_.size();
            double norm = 0.0;
            const size_t base = dW_.size();
            for (size_t j = j1; j <= j2; ++j) {
                double d = std::log2((double)j / (double)i);
                double w = std::exp(-0.5 * d * d / (sigma_ * sigma_)) / (double)j;
                dW_.push_back((float)w);
                norm += w;
            }
            const float inv = (float)(1.0 / norm);
            for (size_t j = base; j < dW_.size(); ++j) dW_[j] *= inv;
        }
        dOff_[directEnd_] = (uint32_t)dW_.size();
        if (directEnd_ >= n) return;

        // log grid: cell k is centred at x0 * 2^(k*h)  (x in bin units)
        const double h  = sigma_ / kCellsPerSigma;
        const double x0 = std::max(1.0, (double)directEnd_ / r2);
        // last cell centre must lie inside the spectrum (<= n-1), otherwise the
        // topmost cells would be empty; bins above it reuse the last cell value
        K_ = (size_t)std::floor(std::log2((double)(n - 1) / x0) / h) + 1;
        if (K_ < 2) K_ = 2;

        jLo_.resize(K_); jHi_.resize(K_); fLo_.resize(K_); fHi_.resize(K_); invW_.resize(K_);
        const double half = std::exp2(0.5 * h);
        auto locate = [&](double x, uint32_t& j, double& f) {
            x = std::clamp(x, 0.5, (double)n - 0.5);
            size_t jj = std::min<size_t>(n - 1, (size_t)std::floor(x + 0.5));
            j = (uint32_t)jj;
            f = x - ((double)jj - 0.5);
        };
        for (size_t k = 0; k < K_; ++k) {
            const double xc = x0 * std::exp2((double)k * h);
            const double xl = std::clamp(xc / half, 0.5, (double)n - 0.5);
            const double xh = std::clamp(xc * half, 0.5, (double)n - 0.5);
            locate(xl, jLo_[k], fLo_[k]);
            locate(xh, jHi_[k], fHi_[k]);
            invW_[k] = 1.0 / std::max(xh - xl, 1e-9);
        }

        const int R = (int)std::ceil(kRadius * kCellsPerSigma);
        g_.resize(2 * R + 1);
        for (int m = -R; m <= R; ++m)
            g_[m + R] = std::exp(-0.5 * (double)(m * m) / (kCellsPerSigma * kCellsPerSigma));
        invG_.resize(K_);
        for (int k = 0; k < (int)K_; ++k) {
            double s = 0.0;
            for (int m = std::max(-R, -k); m <= std::min(R, (int)K_ - 1 - k); ++m)
                s += g_[m + R];
            invG_[k] = 1.0 / s;
        }

        const size_t cnt = n - directEnd_;
        ik_.resize(cnt);
        it_.resize(cnt);
        for (size_t i = directEnd_; i < n; ++i) {
            double u = std::log2((double)i / x0) / h;
            if (u < 0.0) u = 0.0;
            size_t k = (size_t)std::floor(u);
            double t = u - (double)k;
            if (k >= K_ - 1) {
                k = K_ - 2;
                t = 1.0;
            }
            ik_[i - directEnd_] = (uint32_t)k;
            it_[i - directEnd_] = (float)t;
        }

        S_.assign(n + 1, 0.0);
        c_.assign(K_, 0.0);
        sm_.assign(K_, 0.0);
    }
};
