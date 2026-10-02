# AudioFFT (patched)

Bundled copy of [AudioFFT](https://github.com/HiFi-LoFi/AudioFFT) by HiFi-LoFi
(MIT license, original copyright headers kept). This is **not** the pristine
upstream version.

## What is patched

- **Double-precision overloads** added to the public API:

  ```cpp
  void fft(const double* data, double* re, double* im);
  void ifft(double* data, const double* re, const double* im);
  ```

  Implemented via two shared template helpers (`fftT` / `ifftT`) in the Ooura
  backend, plus matching virtual methods in `detail::AudioFFTImpl` and
  forwarding wrappers in `AudioFFT`. The existing `float` API is unchanged
  and uses the same code path.

- Only the built-in **Ooura** backend is patched. Do not define
  `AUDIOFFT_FFTW3` or `AUDIOFFT_APPLE_ACCELERATE`.

## Why

- The minimum-phase (cepstrum) IR calculation needs `double` precision. The
  Ooura backend already computes in `double` internally, so the overloads
  simply avoid the float round-trip at the interface.
- AudioFFT keeps all state inside the object (no global planner), so several
  plugin instances can run in one host process without any locking. This
  replaced FFTW3, whose planner is not thread-safe, and also removes the GPL
  dependency.

## Usage notes

- Size must be a power of 2. This is only `assert`ed, so release builds do
  not check it.
- Real input, split-complex output with `size/2 + 1` bins (DC to Nyquist).
- `fft()` is unnormalized, `ifft()` is already scaled (`ifft(fft(x)) == x`).
- `init()` allocates; `fft()` / `ifft()` do not (realtime-safe after init).
- One object per thread: an object holds an internal work buffer and must
  not be used concurrently.
