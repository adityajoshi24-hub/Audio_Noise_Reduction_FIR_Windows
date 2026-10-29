
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import resample_poly
import soundfile as sf

# ============================================================
# DTSP MICRO PROJECT
# Audio Noise Reduction Using FIR Window Techniques
# ============================================================

# 1. PARAMETERS
fs = 16000                 # Sampling frequency: 16 kHz
duration = 5               # Signal duration in seconds
filter_order = 100         # FIR filter order
num_taps = filter_order + 1
cutoff = 3400              # Low-pass cutoff frequency in Hz
desired_input_snr = 5      # Desired input SNR in dB

# Set True to use your own WAV recording
USE_AUDIO_FILE = False
AUDIO_FILE = "speech.wav"

# 2. LOAD OR GENERATE CLEAN SIGNAL
if USE_AUDIO_FILE:
    clean, original_fs = sf.read(AUDIO_FILE)

    # Convert stereo to mono
    if clean.ndim == 2:
        clean = np.mean(clean, axis=1)

    # Resample to 16 kHz if required
    if original_fs != fs:
        from math import gcd
        g = gcd(fs, original_fs)
        clean = resample_poly(
            clean, fs // g, original_fs // g
        )

    clean = clean.astype(float)
    clean = clean / (np.max(np.abs(clean)) + 1e-12)

else:
    # Synthetic speech-like test signal
    # This is NOT a real speech recording
    t = np.arange(int(fs * duration)) / fs

    envelope = (
        0.65 + 0.25 * np.sin(2 * np.pi * 2 * t)
    )

    clean = envelope * (
        0.50 * np.sin(2 * np.pi * 180 * t)
        + 0.30 * np.sin(2 * np.pi * 450 * t)
        + 0.20 * np.sin(2 * np.pi * 900 * t)
        + 0.12 * np.sin(2 * np.pi * 1800 * t)
        + 0.08 * np.sin(2 * np.pi * 2800 * t)
    )

clean = clean - np.mean(clean)
clean = clean / (np.max(np.abs(clean)) + 1e-12)

# 3. GENERATE HIGH-FREQUENCY NOISE
rng = np.random.default_rng(42)
white_noise = rng.standard_normal(len(clean))

# Noise mainly between 4.2 kHz and 7.5 kHz
sos = signal.butter(
    4, [4200, 7500],
    btype="bandpass", fs=fs, output="sos"
)

noise = signal.sosfiltfilt(sos, white_noise)
noise = noise - np.mean(noise)

# Set noise level to achieve desired input SNR
signal_power = np.mean(clean**2)
noise_power = np.mean(noise**2)

noise = noise * np.sqrt(
    signal_power / (noise_power * 10**(desired_input_snr / 10))
)

noisy = clean + noise

# 4. DESIGN FIR FILTERS USING WINDOW TECHNIQUES
windows = {
    "Hamming": "hamming",
    "Hann": "hann",
    "Blackman": "blackman"
}

filtered_signals = {}
coefficients = {}

for name, window in windows.items():

    # FIR low-pass filter using the window method
    h = signal.firwin(
        num_taps,
        cutoff,
        window=window,
        pass_zero="lowpass",
        fs=fs
    )

    coefficients[name] = h

    # Apply FIR filter
    # Forward-backward filtering avoids phase delay
    y = signal.filtfilt(h, [1.0], noisy)

    filtered_signals[name] = y

# 5. PERFORMANCE MEASURES
def calculate_snr(reference, test_signal):
    error = reference - test_signal

    return 10 * np.log10(
        (np.sum(reference**2) + 1e-15) /
        (np.sum(error**2) + 1e-15)
    )

input_snr = calculate_snr(clean, noisy)

print("\n========== PERFORMANCE RESULTS ==========")
print(f"Input SNR: {input_snr:.2f} dB")
print("\nWindow       Output SNR    SNR Improvement    RMSE")
print("-" * 57)

results = {}

for name, output in filtered_signals.items():

    output_snr = calculate_snr(clean, output)
    improvement = output_snr - input_snr
    rmse = np.sqrt(np.mean((clean - output)**2))

    results[name] = {
        "Output SNR": output_snr,
        "SNR Improvement": improvement,
        "RMSE": rmse
    }

    print(
        f"{name:10s}  {output_snr:8.2f} dB"
        f"    {improvement:8.2f} dB"
        f"      {rmse:.5f}"
    )

best_window = max(
    results, key=lambda name: results[name]["Output SNR"]
)

print(f"\nBest window by output SNR: {best_window}")

# 6. TIME-DOMAIN WAVEFORMS
t = np.arange(len(clean)) / fs
plot_duration = min(0.04, len(clean) / fs)
samples = int(plot_duration * fs)

fig, axes = plt.subplots(3, 1, figsize=(12, 9))

axes[0].plot(t[:samples], clean[:samples])
axes[0].set_title("Original Clean Signal")
axes[0].set_ylabel("Amplitude")
axes[0].grid(True)

axes[1].plot(t[:samples], noisy[:samples])
axes[1].set_title("Noisy Input Signal")
axes[1].set_ylabel("Amplitude")
axes[1].grid(True)

axes[2].plot(t[:samples], filtered_signals[best_window][:samples])
axes[2].set_title(f"Filtered Output ({best_window} Window)")
axes[2].set_xlabel("Time (seconds)")
axes[2].set_ylabel("Amplitude")
axes[2].grid(True)

plt.tight_layout()
plt.show()

# 7. FREQUENCY SPECTRUM COMPARISON
def plot_spectrum(x, label, fs):
    n = len(x)
    spectrum = np.abs(np.fft.rfft(x)) / n
    frequencies = np.fft.rfftfreq(n, d=1/fs)

    # Avoid log of zero
    spectrum_db = 20 * np.log10(spectrum + 1e-12)

    plt.plot(frequencies, spectrum_db, label=label)

plt.figure(figsize=(12, 6))

plot_spectrum(clean, "Clean", fs)
plot_spectrum(noisy, "Noisy", fs)

for name, output in filtered_signals.items():
    plot_spectrum(output, name, fs)

plt.title("Frequency Spectrum Comparison")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.xlim(0, fs / 2)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 8. FILTER FREQUENCY RESPONSES
plt.figure(figsize=(12, 6))

for name, h in coefficients.items():
    f, H = signal.freqz(h, worN=4096, fs=fs)

    magnitude_db = 20 * np.log10(np.abs(H) + 1e-12)

    plt.plot(f, magnitude_db, label=name)

plt.title("FIR Filter Frequency Responses")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.xlim(0, fs / 2)
plt.ylim(-100, 5)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 9. BAR CHART: SNR IMPROVEMENT
names = list(results.keys())
improvements = [
    results[name]["SNR Improvement"] for name in names
]

plt.figure(figsize=(8, 5))
plt.bar(names, improvements)
plt.title("SNR Improvement for Different FIR Windows")
plt.xlabel("Window Technique")
plt.ylabel("SNR Improvement (dB)")
plt.grid(axis="y")
plt.tight_layout()
plt.show()

# 10. SAVE FILTERED AUDIO
for name, output in filtered_signals.items():
    filename = f"filtered_{name.lower()}.wav"

    sf.write(filename, output, fs)
    print(f"Saved: {filename}")

print("\nSimulation completed successfully.")
