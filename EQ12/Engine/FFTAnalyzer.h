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
 * Windowed (Hann), overlapping FFT analysis via AudioFFT using a ring-buffer
 * fifo and hop-size-driven processing. Produces per-bin magnitude in dB
 * with asymmetric attack/release smoothing, exposed through a lock-free
 * double-buffered read/write index for safe access from another thread.
 *
 * AudioFFT keeps all its state inside the object (no global planner),
 * so no locking is needed between plugin instances. The object is NOT
 * safe for concurrent use: processBlock() must only be called from one
 * thread (the audio thread).
****************************************************************/

#pragma once

#include "AudioFFT.h"
#include <cmath>
#include <atomic>
#include <cstring>


class FFTAnalyzer {
public:
    FFTAnalyzer() = default;

    FFTAnalyzer(const FFTAnalyzer&) = delete;
    FFTAnalyzer& operator=(const FFTAnalyzer&) = delete;

    ~FFTAnalyzer() {
        cleanup();
    }

    // Allocates. Call from a non-realtime thread. fft_size must be a power of 2.
    void init(int fft_size, float sr) {
        cleanup();

        // AudioFFT only asserts on the size (debug builds), so guard here.
        if (fft_size < 8 || (fft_size & (fft_size - 1)) != 0)
            return;

        N = fft_size;
        bins = fft_size / 2;
        sample_rate = sr;

        fifo_pos = 0;
        hop_size = N / 2;
        samples_since_last_fft = 0;

        write_index = 0;
        read_index  = 1;
        buffer_ready.store(false, std::memory_order_relaxed);

        fft.init(N);

        in     = new float[N]();
        re     = new float[bins + 1]();   // == AudioFFT::ComplexSize(N)
        im     = new float[bins + 1]();
        window = new float[N]();
        fifo   = new float[N]();
        smooth = new float[bins]();

        mags[0] = new float[bins]();
        mags[1] = new float[bins]();

        build_hann(window, N);

        float window_gain = compute_window_gain(window, N);
        norm_factor = (2.0f / (N * window_gain));

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

        delete[] in;
        delete[] re;
        delete[] im;
        delete[] window;
        delete[] fifo;
        delete[] smooth;
        delete[] mags[0];
        delete[] mags[1];

        in = re = im = window = fifo = smooth = nullptr;
        mags[0] = mags[1] = nullptr;

        initialized = false;
    }

private:
    int N = 0, bins = 0;
    float dc_x1 = 0.0f, dc_y1 = 0.0f;
    float sample_rate = 0.0f;

    audiofft::AudioFFT fft;

    float* in = nullptr;
    float* re = nullptr;
    float* im = nullptr;
    float* window = nullptr;
    float* fifo = nullptr;
    float* smooth = nullptr;

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

        // forward transform: unnormalized, split-complex, bins 0..N/2
        fft.fft(in, re, im);

        float* write_buf = mags[write_index];

        for (int k = 0; k < bins; ++k) {
            float mag = std::sqrt(re[k] * re[k] + im[k] * im[k]);
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
