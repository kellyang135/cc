# neuroguide/backend/app/services/analysis.py
import numpy as np
import mne
from scipy import signal


class AnalysisService:
    """Service for EEG signal analysis operations."""

    def get_signal_segment(
        self,
        raw: mne.io.Raw,
        channels: list[str] | None = None,
        start_time: float = 0.0,
        end_time: float | None = None,
    ) -> dict:
        """Extract a time segment of signal data."""
        if channels is None:
            channels = raw.info['ch_names'][:10]

        available = set(raw.info['ch_names'])
        channels = [ch for ch in channels if ch in available]

        if not channels:
            raise ValueError("No valid channels specified")

        if end_time is None:
            end_time = raw.times[-1]

        picks = mne.pick_channels(raw.info['ch_names'], include=channels)
        start_sample = int(start_time * raw.info['sfreq'])
        end_sample = int(end_time * raw.info['sfreq'])

        data = raw.get_data(picks=picks, start=start_sample, stop=end_sample)
        times = raw.times[start_sample:end_sample]

        return {
            "data": data.tolist(),
            "times": times.tolist(),
            "channels": channels,
            "sample_rate": raw.info['sfreq'],
        }

    def apply_filter(
        self,
        raw: mne.io.Raw,
        low_freq: float | None = None,
        high_freq: float | None = None,
    ) -> mne.io.Raw:
        """Apply bandpass filter to raw data."""
        raw_filtered = raw.copy()
        raw_filtered.filter(l_freq=low_freq, h_freq=high_freq, verbose=False)
        return raw_filtered

    def compute_power_spectrum(
        self,
        raw: mne.io.Raw,
        channel: str,
        fmin: float = 0.5,
        fmax: float = 50.0,
    ) -> dict:
        """Compute power spectral density for a channel."""
        picks = mne.pick_channels(raw.info['ch_names'], include=[channel])
        if len(picks) == 0:
            raise ValueError(f"Channel {channel} not found")

        spectrum = raw.compute_psd(method='welch', picks=picks, fmin=fmin, fmax=fmax, verbose=False)
        freqs = spectrum.freqs
        power = spectrum.get_data()[0]

        return {
            "frequencies": freqs.tolist(),
            "power": power.tolist(),
            "channel": channel,
        }

    def compute_spectrogram(
        self,
        raw: mne.io.Raw,
        channel: str,
        start_time: float = 0.0,
        end_time: float | None = None,
    ) -> dict:
        """Compute spectrogram for a channel."""
        picks = mne.pick_channels(raw.info['ch_names'], include=[channel])
        if len(picks) == 0:
            raise ValueError(f"Channel {channel} not found")

        if end_time is None:
            end_time = raw.times[-1]

        start_sample = int(start_time * raw.info['sfreq'])
        end_sample = int(end_time * raw.info['sfreq'])
        data = raw.get_data(picks=picks, start=start_sample, stop=end_sample)[0]

        fs = raw.info['sfreq']
        nperseg = int(fs * 2)
        noverlap = int(nperseg * 0.75)

        f, t, Sxx = signal.spectrogram(data, fs=fs, nperseg=nperseg, noverlap=noverlap, scaling='density')
        freq_mask = f <= 50

        return {
            "times": (t + start_time).tolist(),
            "frequencies": f[freq_mask].tolist(),
            "power": Sxx[freq_mask, :].tolist(),
            "channel": channel,
        }
