/*
 * Convolver.h
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Stereo partitioned convolver:
 *
 *   Convolver  — 128 samples latency, minimum-phase, host-compensated.
 * 
 * Copyright (C) 2026 brummer <brummer@web.de>
 */

#pragma once

#include <array>
#include <atomic>
#include <cstring>

#ifdef PFFFT_SUPPORT
#include "../pffft/pffft.h"
#else
#include <fftw3.h>
#endif

// ============================================================================
// Convolver — 128 samples latency, minimum-phase
// ============================================================================

class Convolver {
public:
    Convolver() {
#ifdef PFFFT_SUPPORT
        setup = pffft_new_setup(static_cast<int>(FFT_SIZE), PFFFT_REAL);

        buildIn   = static_cast<float*>(
            pffft_aligned_malloc(sizeof(float) * FFT_SIZE));
        buildFreq = static_cast<float*>(
            pffft_aligned_malloc(sizeof(float) * FFT_SIZE));
#else
        buildIn   = (double*)      fftw_malloc(sizeof(double)       * FFT_SIZE);
        buildFreq = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * NUM_BINS);
        buildPlan = fftw_plan_dft_r2c_1d(FFT_SIZE, buildIn, buildFreq, FFTW_ESTIMATE);
#endif
        init(chL);
        init(chR);
    }

    ~Convolver() {
        destroy(chL);
        destroy(chR);
        delete activeIR.load();
        delete pendingIR.load();
        delete trash;

#ifdef PFFFT_SUPPORT
        pffft_aligned_free(buildIn);
        pffft_aligned_free(buildFreq);
        pffft_destroy_setup(setup);
#else
        fftw_destroy_plan(buildPlan);
        fftw_free(buildIn);
        fftw_free(buildFreq);
#endif
    }

    void setBypass(int bp) { bypass = bp; }

    void setIR(const double* irL, const double* irR) {
        delete trash;
        trash = nullptr;

        IRData* ir = new IRData();
        build(ir->H_L, irL);
        build(ir->H_R, irR);

        IRData* old = pendingIR.exchange(ir);
        delete old;
    }

    void reset() {
        resetChannel(chL);
        resetChannel(chR);
    }

    void process(size_t n, const float* inL, const float* inR,
                           float* outL, float* outR) {
        swapIR();

        IRData* ir = activeIR.load();
        if (!ir) {
            if (outL != inL) std::memcpy(outL, inL, n * sizeof(float));
            if (outR != inR) std::memcpy(outR, inR, n * sizeof(float));
            return;
        }

        processChannel(chL, ir->H_L, n, inL, outL);
        processChannel(chR, ir->H_R, n, inR, outR);
    }

    size_t getLatency() const { return PART_SIZE; }

private:

    static constexpr size_t IR_LENGTH = 4096;
    static constexpr size_t PART_SIZE = 128;
    static constexpr size_t FFT_SIZE  = PART_SIZE * 2;
    static constexpr size_t NUM_BINS  = FFT_SIZE / 2 + 1;

    static constexpr size_t NUM_PARTS        = IR_LENGTH / PART_SIZE;  // 32
    static constexpr size_t OUTPUT_FIFO_SIZE = PART_SIZE * 8;

    int bypass = 0;

#ifdef PFFFT_SUPPORT
    PFFFT_Setup* setup = nullptr;
    float* buildIn   = nullptr;
    float* buildFreq = nullptr;
#else
    double*       buildIn   = nullptr;
    fftw_complex* buildFreq = nullptr;
    fftw_plan     buildPlan = nullptr;
#endif

#ifdef PFFFT_SUPPORT
    // PFFFT's native frequency-domain representation contains FFT_SIZE
    // floats. It must not be manually interpreted as ordinary complex bins
    // when using pffft_zconvolve_accumulate().
    using Spectrum = std::array<float, FFT_SIZE>;
#else
    struct Complex { double re = 0.0, im = 0.0; };
    using Spectrum = std::array<Complex, NUM_BINS>;
#endif
    using Part     = std::array<Spectrum, NUM_PARTS>;

    struct IRData { Part H_L, H_R; };

    IRData* trash = nullptr;

    struct Channel {
        std::array<float,  PART_SIZE> dryDelay{};
        size_t dryIdx = 0;
        std::array<float,  PART_SIZE> inFifo{};
        size_t inFill = 0;
        std::array<float,  OUTPUT_FIFO_SIZE> outFifo{};
        size_t outRead = 0, outWrite = 0, available = 0;
        std::array<Spectrum, NUM_PARTS> Xhistory{};
        size_t historyPos = 0;
        std::array<double, PART_SIZE> overlap{};

#ifdef PFFFT_SUPPORT
        float* fftIn    = nullptr;
        float* fftOut   = nullptr;
        float* fftFreq  = nullptr;
        float* fftAccum = nullptr;
#else
        double*       fftIn    = nullptr;
        double*       fftOut   = nullptr;
        fftw_complex* fftFreq  = nullptr;
        fftw_complex* fftAccum = nullptr;
        fftw_plan     planFwd  = nullptr;
        fftw_plan     planInv  = nullptr;
#endif
    };

    Channel chL, chR;
    std::atomic<IRData*> activeIR {nullptr};
    std::atomic<IRData*> pendingIR{nullptr};

    void resetChannel(Channel& ch) {
        ch.dryDelay.fill(0.0f);
        ch.dryIdx = 0;
        ch.inFifo.fill(0.0f);
        ch.inFill = 0;
        ch.outFifo.fill(0.0f);
        ch.outRead = ch.outWrite = ch.available = 0;
        ch.overlap.fill(0.0);
        ch.historyPos = 0;
        for (auto& s : ch.Xhistory)
#ifdef PFFFT_SUPPORT
            s.fill(0.0f);
#else
            for (auto& b : s)
                b = {0.0, 0.0};
#endif
    }

    void init(Channel& ch) {
#ifdef PFFFT_SUPPORT
        ch.fftIn = static_cast<float*>(
            pffft_aligned_malloc(sizeof(float) * FFT_SIZE));
        ch.fftOut = static_cast<float*>(
            pffft_aligned_malloc(sizeof(float) * FFT_SIZE));
        ch.fftFreq = static_cast<float*>(
            pffft_aligned_malloc(sizeof(float) * FFT_SIZE));
        ch.fftAccum = static_cast<float*>(
            pffft_aligned_malloc(sizeof(float) * FFT_SIZE));
#else
        ch.fftIn    = (double*)      fftw_malloc(sizeof(double)       * FFT_SIZE);
        ch.fftOut   = (double*)      fftw_malloc(sizeof(double)       * FFT_SIZE);
        ch.fftFreq  = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * NUM_BINS);
        ch.fftAccum = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * NUM_BINS);
        ch.planFwd  = fftw_plan_dft_r2c_1d(FFT_SIZE, ch.fftIn,    ch.fftFreq,  FFTW_ESTIMATE);
        ch.planInv  = fftw_plan_dft_c2r_1d(FFT_SIZE, ch.fftAccum, ch.fftOut,   FFTW_ESTIMATE);
#endif
        resetChannel(ch);
    }

    void destroy(Channel& ch) {
#ifdef PFFFT_SUPPORT
        pffft_aligned_free(ch.fftIn);
        pffft_aligned_free(ch.fftOut);
        pffft_aligned_free(ch.fftFreq);
        pffft_aligned_free(ch.fftAccum);
#else
        if (ch.planFwd) fftw_destroy_plan(ch.planFwd);
        if (ch.planInv) fftw_destroy_plan(ch.planInv);
        fftw_free(ch.fftIn);
        fftw_free(ch.fftOut);
        fftw_free(ch.fftFreq);
        fftw_free(ch.fftAccum);
#endif
    }

#ifdef PFFFT_SUPPORT
    void build(Part& H, const double* ir) {
#else
    void build(Part& H, const double* ir) {
        // Reuse permanent buildPlan — no planner access at runtime
#endif
        for (size_t p = 0; p < NUM_PARTS; ++p) {
#ifdef PFFFT_SUPPORT
            std::memset(buildIn, 0, sizeof(float) * FFT_SIZE);
#else
            std::memset(buildIn, 0, sizeof(double) * FFT_SIZE);
#endif
            for (size_t i = 0; i < PART_SIZE; ++i) {
                size_t idx = p * PART_SIZE + i;
#ifdef PFFFT_SUPPORT
                if (idx < IR_LENGTH)
                    buildIn[i] = static_cast<float>(ir[idx]);
#else
                if (idx < IR_LENGTH) buildIn[i] = ir[idx];
#endif
            }
#ifdef PFFFT_SUPPORT
            pffft_transform(
                setup,
                buildIn,
                buildFreq,
                nullptr,
                PFFFT_FORWARD);

            std::memcpy(
                H[p].data(),
                buildFreq,
                sizeof(float) * FFT_SIZE);
#else
            fftw_execute(buildPlan);
            for (size_t k = 0; k < NUM_BINS; ++k) {
                H[p][k].re = buildFreq[k][0];
                H[p][k].im = buildFreq[k][1];
            }
#endif
        }
    }

    void swapIR() {
        IRData* p = pendingIR.exchange(nullptr);
        if (!p) return;
        trash = activeIR.exchange(p);
    }

    inline float processDry(Channel& ch, float in) {
        float out = ch.dryDelay[ch.dryIdx];
        ch.dryDelay[ch.dryIdx] = in;
        if (++ch.dryIdx >= PART_SIZE) ch.dryIdx = 0;
        return out;
    }

    void processChannel(Channel& ch, const Part& H, size_t n,
                        const float* in, float* out) {
        for (size_t i = 0; i < n; ++i) {
            float wet = popOutput(ch);
            float dry = processDry(ch, in[i]);
            out[i] = bypass ? dry : wet;
            ch.inFifo[ch.inFill++] = in[i];
            if (ch.inFill == PART_SIZE) {
                runBlock(ch, H);
                ch.inFill = 0;
            }
        }
    }

    inline float popOutput(Channel& ch) {
        if (ch.available == 0) return 0.0f;
        float v = ch.outFifo[ch.outRead];
        if (++ch.outRead >= OUTPUT_FIFO_SIZE) ch.outRead = 0;
        ch.available--;
        return v;
    }

    inline void pushOutput(Channel& ch, float v) {
        if (ch.available >= OUTPUT_FIFO_SIZE) return;
        ch.outFifo[ch.outWrite] = v;
        if (++ch.outWrite >= OUTPUT_FIFO_SIZE) ch.outWrite = 0;
        ch.available++;
    }

    void runBlock(Channel& ch, const Part& H) {
#ifdef PFFFT_SUPPORT
        std::memset(ch.fftIn, 0, sizeof(float) * FFT_SIZE);
#else
        std::memset(ch.fftIn, 0, sizeof(double) * FFT_SIZE);
#endif
        for (size_t i = 0; i < PART_SIZE; ++i)
            ch.fftIn[i] = ch.inFifo[i];
#ifdef PFFFT_SUPPORT
        pffft_transform(
            setup,
            ch.fftIn,
            ch.fftFreq,
            nullptr,
            PFFFT_FORWARD);
#else
        fftw_execute(ch.planFwd);
#endif

        ch.historyPos = (ch.historyPos + NUM_PARTS - 1) % NUM_PARTS;
        Spectrum& Xnew = ch.Xhistory[ch.historyPos];
#ifdef PFFFT_SUPPORT
        std::memcpy(
            Xnew.data(),
            ch.fftFreq,
            sizeof(float) * FFT_SIZE);

        std::memset(ch.fftAccum, 0, sizeof(float) * FFT_SIZE);
#else
        for (size_t k = 0; k < NUM_BINS; ++k) {
            Xnew[k].re = ch.fftFreq[k][0];
            Xnew[k].im = ch.fftFreq[k][1];
        }

        std::memset(ch.fftAccum, 0, sizeof(fftw_complex) * NUM_BINS);
#endif
        for (size_t p = 0; p < NUM_PARTS; ++p) {
            const Spectrum& X  = ch.Xhistory[(ch.historyPos + p) % NUM_PARTS];
            const Spectrum& Hp = H[p];
#ifdef PFFFT_SUPPORT
            pffft_zconvolve_accumulate(
                setup,
                X.data(),
                Hp.data(),
                ch.fftAccum,
                1.0f);
#else
            for (size_t k = 0; k < NUM_BINS; ++k) {
                ch.fftAccum[k][0] += X[k].re * Hp[k].re - X[k].im * Hp[k].im;
                ch.fftAccum[k][1] += X[k].re * Hp[k].im + X[k].im * Hp[k].re;
            }
#endif
        }

#ifdef PFFFT_SUPPORT
        pffft_transform(
            setup,
            ch.fftAccum,
            ch.fftOut,
            nullptr,
            PFFFT_BACKWARD);

        constexpr double scale = 1.0 / static_cast<double>(FFT_SIZE);
#else
        fftw_execute(ch.planInv);
        constexpr double scale = 1.0 / FFT_SIZE;
#endif
        for (size_t i = 0; i < PART_SIZE; ++i) {
            float v = (float)(ch.fftOut[i] * scale + ch.overlap[i]);
#ifdef PFFFT_SUPPORT
            ch.overlap[i] =
                static_cast<double>(ch.fftOut[i + PART_SIZE]) * scale;
#else
            ch.overlap[i] = ch.fftOut[i + PART_SIZE] * scale;
#endif
            pushOutput(ch, v);
        }
    }
};
