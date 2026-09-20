
/*
 * FFTAnalyzer.h
 *
 * SPDX-License-Identifier:  BSD-3-Clause
 *
 * Copyright (C) 2025 brummer <brummer@web.de>
 */

/****************************************************************
 * @file FFTAnalyzer.h
 * @brief Real-time FFT-based spectrum analyzer with smoothed magnitude output.
 *
 * Windowed (Hann), overlapping FFT analysis via FFTW using a ring-buffer
 * fifo and hop-size-driven processing. Produces per-bin magnitude in dB
 * with asymmetric attack/release smoothing, exposed through a lock-free
 * double-buffered read/write index for safe access from another thread.
****************************************************************/

#pragma once

#ifdef PFFFT_SUPPORT
#include "../pffft/pffft.h"
#include <stdexcept>
#else
#include <fftw3.h>
#endif
#include <cmath>
#include <atomic>
#include <cstring>


class FFTAnalyzer {
public:
    FFTAnalyzer() = default;

    ~FFTAnalyzer() {
        cleanup();
    }

    void init(int fft_size, float sr) {
        cleanup(); 

        N = fft_size;
        bins = fft_size / 2;
        sample_rate = sr;

        fifo_pos = 0;
        hop_size = N / 2;
        samples_since_last_fft = 0;

        write_index = 0;
        read_index  = 1;
        buffer_ready.store(false, std::memory_order_relaxed);
#ifdef PFFFT_SUPPORT
        setup = pffft_new_setup(N, PFFFT_REAL);
        if (!setup)
            throw std::invalid_argument(
                "FFT size is not supported by PFFFT");

        // PFFFT requires SIMD-compatible alignment for transform buffers.
        // For a real transform, both buffers contain N floats.
        in = static_cast<float*>(
            pffft_aligned_malloc(sizeof(float) * N));
        out = static_cast<float*>(
            pffft_aligned_malloc(sizeof(float) * N));

        if (!in || !out) {
            pffft_aligned_free(in);
            pffft_aligned_free(out);
            in = nullptr;
            out = nullptr;
            pffft_destroy_setup(setup);
            setup = nullptr;
            throw std::bad_alloc();
        }
#else
        in     = (float*)fftwf_malloc(sizeof(float) * N);
        out    = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (bins + 1));
#endif
        window = new float[N];
        fifo   = new float[N]();
        smooth = new float[bins];

        mags[0] = new float[bins];
        mags[1] = new float[bins];

        std::memset(in, 0,  N * sizeof(float));
        std::memset(out, 0,  N * sizeof(float));
        std::memset(window, 0, N * sizeof(float));
        std::memset(fifo, 0, N * sizeof(float));
        std::memset(smooth, 0, bins * sizeof(float));

        std::memset(mags[0], 0, bins * sizeof(float));
        std::memset(mags[1], 0, bins * sizeof(float));

        build_hann(window, N);

        float window_gain = compute_window_gain(window, N);
        norm_factor = (2.0f / (N * window_gain));

#ifndef PFFFT_SUPPORT
        plan = fftwf_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);
#endif
        for (int i = 0; i < bins; ++i) smooth[i] = -90.0f;

        attack  = 0.6f;
        release = 0.05f;
        initialized = true;
    }

    bool isInitialized() const {
        return initialized;
    }

    void reset() {
        if (!initialized) return;

        std::memset(fifo, 0, sizeof(float) * N);
        fifo_pos = 0;
        samples_since_last_fft = 0;
        dc_x1 = 0.0f;
        dc_y1 = 0.0f;

        for (int i = 0; i < bins; ++i)
            smooth[i] = -90.0f;

        buffer_ready.store(false, std::memory_order_release);
    }

    void processBlock(const float* input, int n_samples) {
        if (!initialized) return;

        for (int i = 0; i < n_samples; ++i) {

            float v = input[i];
            if (!std::isfinite(v)) v = 0.0f;

            float y = v - dc_x1 + 0.995f * dc_y1;
            dc_x1 = v;
            dc_y1 = y;

            fifo[fifo_pos++] = y;
            if (fifo_pos >= N)
                fifo_pos = 0;

            samples_since_last_fft++;

            while (samples_since_last_fft >= hop_size) {
                samples_since_last_fft -= hop_size;
                processFFT();
            }
        }
    }

    const float* getMagnitudes() const {
        return mags[read_index];
    }

    bool hasNewData() const {
        return buffer_ready.load(std::memory_order_acquire);
    }

    void clearFlag() {
        buffer_ready.store(false, std::memory_order_release);
    }

    int getBins() const { return bins; }

    void cleanup() {
        if (!initialized) return;
#ifdef PFFFT_SUPPORT
        pffft_aligned_free(in);
        pffft_aligned_free(out);
        pffft_destroy_setup(setup);
#else
        fftwf_destroy_plan(plan);
        fftwf_free(in);
        fftwf_free(out);
#endif
        delete[] window;
        delete[] fifo;
        delete[] smooth;
        delete[] mags[0];
        delete[] mags[1];
#ifdef PFFFT_SUPPORT
        in = nullptr;
        out = nullptr;
        setup = nullptr;
        window = nullptr;
        fifo = nullptr;
        smooth = nullptr;
        mags[0] = nullptr;
        mags[1] = nullptr;
#endif
        initialized = false;
    }

private:
    int N = 0, bins = 0;
    float dc_x1 = 0.0f, dc_y1 = 0.0f;
    float sample_rate = 0.0f;

    float* in = nullptr;
    float* window = nullptr;
    float* fifo = nullptr;
    float* smooth = nullptr;
#ifdef PFFFT_SUPPORT
    float* out = nullptr;
    PFFFT_Setup* setup = nullptr;
#else
    fftwf_complex* out = nullptr;
    fftwf_plan plan = nullptr;
#endif
    float* mags[2] = {nullptr, nullptr};

    int write_index = 0;
    int read_index  = 1;

    std::atomic<bool> buffer_ready {false};

    float norm_factor = 1.0f;

    int fifo_pos = 0;
    int hop_size = 0;
    int samples_since_last_fft = 0;

    float attack = 0.6f;
    float release = 0.05f;

    bool initialized = false;

private:

    void build_hann(float* w, int N) {
        for (int i = 0; i < N; ++i) {
            w[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (N - 1)));
        }
    }

    void build_hamming_window(float *w, int N) {
        for (int i = 0; i < N; ++i) {
            w[i] = 0.54 - 0.46 * cosf(2.0 * M_PI * i / (N - 1));
        }
    }

    void build_blackman_harris(float* w, int N) {
        const float a0 = 0.35875f;
        const float a1 = 0.48829f;
        const float a2 = 0.14128f;
        const float a3 = 0.01168f;

        for (int i = 0; i < N; ++i) {
            float arg = (2.0f * (float)M_PI * i) / (N - 1);
            w[i] = a0 - a1 * cosf(arg) + a2 * cosf(2.0f * arg) - a3 * cosf(3.0f * arg);
        }
    }

    void build_blackman_harris_3_term(float* w, int N) {
        const float a0 = 0.4243801;
        const float a1 = 0.4973406;
        const float a2 = 0.0782793;
        for (int i = 0; i < N; ++i) {
            float angle1 = (2.0 * M_PI * i) / (N - 1);
            float angle2 = (4.0 * M_PI * i) / (N - 1);
            w[i] = a0 - a1 * cosf(angle1) + a2 * cosf(angle2);
        }
    }

    float compute_window_gain(const float* w, int N) {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i)
            sum += w[i];
        return sum / (float)N;
    }

    void processFFT() {
        int idx = fifo_pos - N;
        if (idx < 0) idx += N;
        for (int j = 0; j < N; ++j) {
            float s = fifo[idx];
            in[j] = s * window[j];

            idx++;
            if (idx >= N)
                idx = 0;
        }
#ifdef PFFFT_SUPPORT
        // The ordered real-transform layout is:
        //
        //   out[0]       = DC real component
        //   out[1]       = Nyquist real component
        //   out[2*k]     = real component of bin k
        //   out[2*k + 1] = imaginary component of bin k
        //
        // Only bins [0, N / 2) are processed, matching the original
        // FFTW implementation.
        pffft_transform_ordered(
            setup,
            in,
            out,
            nullptr,
            PFFFT_FORWARD);
#else
        fftwf_execute(plan);
#endif
        float* write_buf = mags[write_index];

        for (int k = 0; k < bins; ++k) {
#ifdef PFFFT_SUPPORT
            float re;
            float im;

            if (k == 0) {
                re = out[0];
                im = 0.0f;
            } else {
                re = out[2 * k];
                im = out[2 * k + 1];
            }
#else
            float re = out[k][0];
            float im = out[k][1];
#endif
            float mag = std::sqrt(re * re + im * im);
            mag *= norm_factor;

            if (!std::isfinite(mag) || mag <= 0.0f)
                mag = 1e-20f;

            float db = 20.0f * std::log10(mag);

            float prev = smooth[k];
            float t = (float)k / (float)bins;
            float rel = release * (0.5f + t);

            float coeff = (db > prev) ? attack : rel;
            float smoothed = prev + coeff * (db - prev);

            smooth[k] = smoothed;
            write_buf[k] = smoothed;
        }

        // swap
        int new_read = write_index;
        write_index = read_index;
        read_index  = new_read;

        buffer_ready.store(true, std::memory_order_release);
    }
};
