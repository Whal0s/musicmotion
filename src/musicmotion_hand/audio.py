from __future__ import annotations

import threading
from typing import Dict, List, Optional

import numpy as np
import sounddevice as sd


# Musical scale definitions (note frequencies in Hz)
# Each scale is a list of MIDI note numbers
SCALES: Dict[str, List[int]] = {
    "c_major": [60, 62, 64, 65, 67, 69, 71, 72],  # C4 to C5
    "c_minor": [60, 62, 63, 65, 67, 68, 70, 72],
    "pentatonic": [60, 62, 64, 67, 69, 72],  # C pentatonic
    "blues": [60, 63, 65, 66, 67, 70, 72],
    "chromatic": list(range(60, 73)),  # C4 to C5 chromatic
}


def midi_to_freq(midi_note: int) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


class ToneGenerator:
    """
    Real-time tone generator for hand position sonification.
    
    Generates sine wave tones based on configurable musical scales.
    """

    def __init__(
        self,
        scale: str = "pentatonic",
        sample_rate: int = 44100,
        volume: float = 0.3,
    ) -> None:
        """
        Initialize the tone generator.
        
        Args:
            scale: Name of the scale to use (from SCALES dict)
            sample_rate: Audio sample rate in Hz
            volume: Volume level (0.0 to 1.0)
        """
        if scale not in SCALES:
            raise ValueError(f"Unknown scale '{scale}'. Available: {list(SCALES.keys())}")
        
        self.scale_name = scale
        self.scale_notes = SCALES[scale]
        self.sample_rate = sample_rate
        self.volume = max(0.0, min(1.0, volume))
        
        # Audio stream state
        self._stream: Optional[sd.OutputStream] = None
        self._lock = threading.Lock()
        self._current_frequency = 0.0
        self._phase = 0.0
        self._is_active = False

    def start(self) -> None:
        """Start the audio stream."""
        if self._stream is not None:
            return
        
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self._audio_callback,
            blocksize=1024,
        )
        self._stream.start()

    def stop(self) -> None:
        """Stop the audio stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def set_position(self, y_normalized: float, active: bool = True) -> None:
        """
        Set the pitch based on normalized Y position.
        
        Args:
            y_normalized: Vertical position (0.0 = top, 1.0 = bottom)
            active: Whether sound should be playing
        """
        with self._lock:
            self._is_active = active
            if active:
                # Invert y so that higher hand (lower y value) = higher pitch
                inverted_y = 1.0 - y_normalized
                # Map to scale index
                scale_index = int(inverted_y * (len(self.scale_notes) - 1))
                scale_index = max(0, min(len(self.scale_notes) - 1, scale_index))
                midi_note = self.scale_notes[scale_index]
                self._current_frequency = midi_to_freq(midi_note)

    def _audio_callback(self, outdata, frames, time_info, status) -> None:
        """
        Audio callback for sounddevice stream.
        
        Args:
            outdata: Output buffer to fill with audio samples
            frames: Number of frames to generate
            time_info: Timing information from the audio system
            status: Stream status flags indicating errors or warnings
        """
        with self._lock:
            if not self._is_active or self._current_frequency <= 0:
                outdata[:] = 0
                return
            
            # Generate sine wave
            t = (np.arange(frames) + self._phase) / self.sample_rate
            wave = self.volume * np.sin(2 * np.pi * self._current_frequency * t)
            outdata[:, 0] = wave.astype(np.float32)
            
            # Update phase for continuity (use modulo to prevent overflow)
            self._phase = (self._phase + frames) % (self.sample_rate * 1000)

    def __enter__(self) -> "ToneGenerator":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
