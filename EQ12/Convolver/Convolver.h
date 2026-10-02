/*
 * Convolver.h
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Stereo partitioned convolver:
 *
 *   Convolver  — 128 samples latency, minimum-phase, host-compensated.
 *
 * FFT layer: AudioFFT (split-complex, all state inside the object, no global
 * planner -> no locking needed between plugin instances).
 *
 * Copyright (C) 2026 brummer <brummer@web.de>
 */

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstring>

#include "AudioFFT.h"

// ============================================================================
// Convolver — 128 samples latency, minimum-phase
// ============================================================================

class Convolver {
public:
    Convolver() {
        // init() allocates -> constructor only, never in the audio thread
        buildFft.init(FFT_SIZE);
        initChannel(chL);
        initChannel(chR);
    }

    ~Convolver() {
        delete activeIR.load();
        delete pendingIR.load();
        delete trash.load();
    }

    void setBypass(int bp) { bypass = bp; }

    // Must not be called concurrently from several threads (shares buildFft/buildIn).
    void setIR(const double* irL, const double* irR) {
        // Collect the IR the audio thread retired after its last swap.
        delete trash.exchange(nullptr);

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
    static constexpr size_t NUM_BINS  = FFT_SIZE / 2 + 1;   // == AudioFFT::ComplexSize(FFT_SIZE)

    static constexpr size_t NUM_PARTS        = IR_LENGTH / PART_SIZE;  // 32
    static constexpr size_t OUTPUT_FIFO_SIZE = PART_SIZE * 8;

    int bypass = 0;

    // Only used by setIR()/build(), i.e. never from the audio thread.
    audiofft::AudioFFT          buildFft;
    std::array<float, FFT_SIZE> buildIn{};

    // Split-complex spectrum (bins 0..N/2, as AudioFFT delivers it)
    struct Spectrum {
        std::array<float, NUM_BINS> re{};
        std::array<float, NUM_BINS> im{};
    };
    using Part = std::array<Spectrum, NUM_PARTS>;

    struct IRData { Part H_L, H_R; };

    // Slot for the IR retired by the audio thread; freed by setIR()/destructor.
    std::atomic<IRData*> trash {nullptr};

    struct Channel {
        std::array<float,  PART_SIZE> dryDelay{};
        size_t dryIdx = 0;
        std::array<float,  PART_SIZE> inFifo{};
        size_t inFill = 0;
        std::array<float,  OUTPUT_FIFO_SIZE> outFifo{};
        size_t outRead = 0, outWrite = 0, available = 0;
        std::array<Spectrum, NUM_PARTS> Xhistory{};
        size_t historyPos = 0;
        std::array<float, PART_SIZE> overlap{};

        // Upper half of fftIn stays zero forever (zero padding), only the
        // lower half is rewritten per block.
        std::array<float, FFT_SIZE> fftIn{};
        std::array<float, FFT_SIZE> fftOut{};
        std::array<float, NUM_BINS> accRe{};
        std::array<float, NUM_BINS> accIm{};

        audiofft::AudioFFT fft;   // one object per channel: not usable concurrently
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
        ch.overlap.fill(0.0f);
        ch.historyPos = 0;
        for (auto& s : ch.Xhistory) {
            s.re.fill(0.0f);
            s.im.fill(0.0f);
        }
        ch.fftIn.fill(0.0f);
        ch.fftOut.fill(0.0f);
        ch.accRe.fill(0.0f);
        ch.accIm.fill(0.0f);
    }

    void initChannel(Channel& ch) {
        ch.fft.init(FFT_SIZE);
        resetChannel(ch);
    }

    void build(Part& H, const double* ir) {
        for (size_t p = 0; p < NUM_PARTS; ++p) {
            buildIn.fill(0.0f);
            for (size_t i = 0; i < PART_SIZE; ++i) {
                size_t idx = p * PART_SIZE + i;
                if (idx < IR_LENGTH) buildIn[i] = static_cast<float>(ir[idx]);
            }
            // transform straight into the partition's spectrum
            buildFft.fft(buildIn.data(), H[p].re.data(), H[p].im.data());
        }
    }

    void swapIR() {
        IRData* p = pendingIR.exchange(nullptr);
        if (!p) return;
        // Never delete in the audio thread: hand the old IR over to setIR().
        // (setIR() empties this slot before it can post the next pending IR.)
        trash.store(activeIR.exchange(p));
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
        for (size_t i = 0; i < PART_SIZE; ++i)
            ch.fftIn[i] = ch.inFifo[i];

        // forward FFT directly into the newest history slot
        ch.historyPos = (ch.historyPos + NUM_PARTS - 1) % NUM_PARTS;
        Spectrum& Xnew = ch.Xhistory[ch.historyPos];
        ch.fft.fft(ch.fftIn.data(), Xnew.re.data(), Xnew.im.data());

        // frequency-domain multiply-accumulate over all partitions
        float* accRe = ch.accRe.data();
        float* accIm = ch.accIm.data();
        for (size_t k = 0; k < NUM_BINS; ++k) { accRe[k] = 0.0f; accIm[k] = 0.0f; }

        for (size_t p = 0; p < NUM_PARTS; ++p) {
            const Spectrum& X  = ch.Xhistory[(ch.historyPos + p) % NUM_PARTS];
            const Spectrum& Hp = H[p];
            const float* xr = X.re.data();
            const float* xi = X.im.data();
            const float* hr = Hp.re.data();
            const float* hi = Hp.im.data();
            for (size_t k = 0; k < NUM_BINS; ++k) {
                accRe[k] += xr[k] * hr[k] - xi[k] * hi[k];
                accIm[k] += xr[k] * hi[k] + xi[k] * hr[k];
            }
        }

        // AudioFFT::ifft() already scales (no 1/N here)
        ch.fft.ifft(ch.fftOut.data(), accRe, accIm);

        for (size_t i = 0; i < PART_SIZE; ++i) {
            float v = ch.fftOut[i] + ch.overlap[i];
            ch.overlap[i] = ch.fftOut[i + PART_SIZE];
            pushOutput(ch, v);
        }
    }
};
