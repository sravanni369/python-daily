"""Day 10 - Signal Denoising.

Recover a clean signal from noisy measurements and score the result. Two
stdlib-only filters (moving average for gaussian-ish noise, median for spikes)
plus a signal-to-noise ratio so you can prove the denoising actually helped.

This is the fundamental under any "reconstruct a signal from noisy sensor data"
problem. Run: python day10_signal_denoising.py
"""

import math
import random


def moving_average(signal, window=3):
    """Smooth a signal by averaging each point with its neighbours.

    Uses a centered window and shrinks the window at the edges so the output
    stays the same length as the input.
    """
    n = len(signal)
    half = window // 2
    out = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        chunk = signal[lo:hi]
        out.append(sum(chunk) / len(chunk))
    return out


def median_filter(signal, window=3):
    """Replace each point with the median of its window. Kills spikes that a
    moving average would only smear."""
    n = len(signal)
    half = window // 2
    out = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        chunk = sorted(signal[lo:hi])
        m = len(chunk)
        out.append(chunk[m // 2] if m % 2 else (chunk[m // 2 - 1] + chunk[m // 2]) / 2)
    return out


def snr_db(clean, estimate):
    """Signal-to-noise ratio in decibels: how much true signal survives the
    error left after denoising. Higher is better; +3 dB ~ half the error power.
    """
    signal_power = sum(c * c for c in clean)
    error_power = sum((c - e) ** 2 for c, e in zip(clean, estimate))
    if error_power == 0:
        return float("inf")
    return 10 * math.log10(signal_power / error_power)


def add_noise(clean, sigma=0.6, spike_every=17, spike=6.0, seed=10):
    """Corrupt a clean signal with gaussian noise plus occasional spikes,
    mimicking noisy sensor readings."""
    rng = random.Random(seed)
    noisy = []
    for i, c in enumerate(clean):
        v = c + rng.gauss(0, sigma)
        if i % spike_every == 0 and i > 0:
            v += spike * (1 if rng.random() > 0.5 else -1)
        noisy.append(v)
    return noisy


if __name__ == "__main__":
    # a clean signal we pretend the sensor is trying to measure
    clean = [math.sin(i / 6.0) * 3 for i in range(60)]

    noisy = add_noise(clean)
    smoothed = moving_average(noisy, window=5)
    despiked = median_filter(smoothed, window=5)

    print(f"SNR noisy vs clean:    {snr_db(clean, noisy):6.2f} dB")
    print(f"SNR moving-average:    {snr_db(clean, smoothed):6.2f} dB")
    print(f"SNR + median (spikes): {snr_db(clean, despiked):6.2f} dB")

    gain = snr_db(clean, despiked) - snr_db(clean, noisy)
    print(f"improvement:          +{gain:6.2f} dB")
