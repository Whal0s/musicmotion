from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import sounddevice as sd
import queue
import threading

# Allow running without installing the package (repo-local usage).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from musicmotion_hand.detector import HandPositionDetector  # noqa: E402
from musicmotion_hand.types import HandLandmark, HandPosition  # noqa: E402


# --- Tuning knobs (edit these to tweak "two fists" recognition) ---
FIST_SCORE_THRESHOLD = 0.5  # higher = stricter
PROCEED_HOLD_S = 0.2
PROCEED_COOLDOWN_S = 1
# Proceed based on openness being ~0 for both hands.
# If your openness is truly exactly 0.00 when clenched, you can set this to 0.0.
PROCEED_OPENNESS_MAX = 0.03

# --- Metronome tuning knobs ---
METRONOME_ENABLED = True
METRONOME_BEAT_VOLUME = 0.045  # 0..1 (beats 2/3/4)
METRONOME_DOWNBEAT_VOLUME = 0.075  # 0..1 (beat 1)
METRONOME_CLICK_HZ = 1100.0
METRONOME_DOWNBEAT_HZ = 750.0
METRONOME_CLICK_MS = 28.0

# --- Keys (chords) synth knobs ---
KEYS_ENABLED = True
KEYS_VOLUME = 0.16  # 0..1
KEYS_ATTACK_MS = 8.0
KEYS_DECAY_MS = 650.0
KEYS_DURATION_MS = 900.0
KEYS_HOLD_CLOSE_TH = 0.06
KEYS_HOLD_OPEN_TH = 0.16
KEYS_VIBRATO_HZ = 6.0
KEYS_VIBRATO_MAX_CENTS = 35.0
KEYS_VIBRATO_HEIGHT_PX_FOR_MAX = 140.0

# --- Bass synth knobs ---
BASS_ENABLED = True
BASS_VOLUME = 0.18  # 0..1
BASS_ATTACK_MS = 6.0
BASS_DECAY_MS = 380.0
BASS_DURATION_MS = 520.0
BASS_SUB_MIX = 0.35  # 0..1 (adds a sine an octave down)
# More "real synth" tone controls
BASS_HARMONICS = 12
BASS_LPF_HZ = 650.0
BASS_DRIVE = 1.4  # >1 adds saturation

# --- Kit (drums) synth knobs ---
KIT_ENABLED = True
KIT_KICK_VOLUME = 0.22  # 0..1
KIT_SNARE_VOLUME = 0.16  # 0..1
KIT_TRIGGER_CLOSE_TH = 0.05
KIT_TRIGGER_OPEN_TH = 0.12
KIT_TRIGGER_COOLDOWN_S = 0.12

# Less-retro drum tone controls
KICK_DRIVE = 1.6
SNARE_HP_HZ = 700.0
SNARE_LP_HZ = 6500.0
SNARE_TONE_HZ = 190.0

# --- Instrument selection (left hand) tuning ---
INSTR_SELECT_STABLE_FRAMES = 6
INSTR_SELECT_COOLDOWN_S = 0.7


class Metronome:
    def __init__(self, sample_rate: int = 48000, channels: int = 2) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self._bpm = 120.0
        self._beat_samples = int(self.sample_rate * 60.0 / self._bpm)
        self._sample_index = 0
        self._next_click_sample = 0
        self._stream: Optional[sd.OutputStream] = None

        n = max(1, int(self.sample_rate * (METRONOME_CLICK_MS / 1000.0)))
        t = np.arange(n, dtype=np.float32) / float(self.sample_rate)
        env = np.exp(-t * 45.0).astype(np.float32)
        click = (np.sin(2.0 * np.pi * float(METRONOME_CLICK_HZ) * t) * env).astype(np.float32)
        down = (np.sin(2.0 * np.pi * float(METRONOME_DOWNBEAT_HZ) * t) * env).astype(np.float32)
        self._click = click
        self._down = down

        self._events: "queue.SimpleQueue[tuple[str, object]]" = queue.SimpleQueue()
        self._voices: List["_ChordVoice"] = []
        self._sustain: List["_SustainChordVoice"] = []
        self._drums: List["_DrumVoice"] = []
        self._bass: List["_BassVoice"] = []
        self._vib_phase = 0.0
        self._live_vibrato_cents = 0.0
        self._lock = threading.Lock()
        self._playback_enabled = False
        self._loop_layers: List[List[tuple[int, str, object]]] = []  # per-layer events: (sample_offset, kind, payload)

    def set_bpm(self, bpm: int) -> None:
        bpm = int(max(30, min(300, bpm)))
        self._bpm = float(bpm)
        self._beat_samples = max(1, int(self.sample_rate * 60.0 / self._bpm))

    def reset_phase(self) -> None:
        """Reset beat grid so beat 1 happens immediately."""
        with self._lock:
            self._sample_index = 0
            self._next_click_sample = 0

    def start(self) -> None:
        if self._stream is not None:
            return
        self.reset_phase()
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            callback=self._callback,
            blocksize=0,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None

    def _callback(self, outdata, frames, time_info, status) -> None:
        out = np.zeros((frames, self.channels), dtype=np.float32)

        with self._lock:
            live_vib_cents = float(self._live_vibrato_cents)

        with self._lock:
            playback_enabled = self._playback_enabled
            loop_layers = self._loop_layers
            start_idx = int(self._sample_index)

        if METRONOME_ENABLED and (METRONOME_BEAT_VOLUME > 0 or METRONOME_DOWNBEAT_VOLUME > 0):
            start = start_idx
            end = start_idx + frames
            while self._next_click_sample < end:
                off = int(self._next_click_sample - start)
                if off < 0:
                    self._next_click_sample += self._beat_samples
                    continue
                beat_idx = 0
                if self._beat_samples > 0:
                    beat_idx = int(self._next_click_sample // self._beat_samples)
                is_downbeat = (beat_idx % 4) == 0
                vol = float(METRONOME_DOWNBEAT_VOLUME if is_downbeat else METRONOME_BEAT_VOLUME)
                wave = self._down if is_downbeat else self._click
                n = min(len(self._click), frames - off)
                if n > 0:
                    out[off : off + n, 0] += wave[:n] * vol
                    if self.channels > 1:
                        out[off : off + n, 1] += wave[:n] * vol
                self._next_click_sample += self._beat_samples

        # Loop playback: fire recorded events at their sample offsets (once per loop).
        if playback_enabled and loop_layers:
            start = start_idx
            end = start_idx + frames
            for layer in loop_layers:
                for off, kind, payload in layer:
                    if start <= off < end:
                        if kind == "chord":
                            freqs = np.asarray(payload, dtype=np.float64)
                            if freqs.size > 0:
                                self._voices.append(_ChordVoice.from_freqs(freqs, self.sample_rate))
                        elif kind == "chord_on":
                            freqs = np.asarray(payload, dtype=np.float64)
                            if freqs.size > 0:
                                self._sustain.append(_SustainChordVoice.from_freqs(freqs, self.sample_rate))
                        elif kind == "chord_off":
                            if self._sustain:
                                self._sustain[-1].release()
                        elif kind == "bass":
                            self._bass.append(_BassVoice.create(float(payload), self.sample_rate))
                        elif kind == "kick":
                            self._drums.append(_KickVoice.create(self.sample_rate))
                        elif kind == "snare":
                            self._drums.append(_SnareVoice.create(self.sample_rate))

        # Events: chords / bass / drums
        try:
            while True:
                kind, payload = self._events.get_nowait()
                if kind == "chord":
                    freqs = np.asarray(payload, dtype=np.float64)
                    if freqs.size > 0:
                        self._voices.append(_ChordVoice.from_freqs(freqs, self.sample_rate))
                elif kind == "chord_on":
                    freqs = np.asarray(payload, dtype=np.float64)
                    if freqs.size > 0:
                        self._sustain.append(_SustainChordVoice.from_freqs(freqs, self.sample_rate))
                elif kind == "chord_off":
                    if self._sustain:
                        self._sustain[-1].release()
                elif kind == "bass":
                    freq = float(payload)
                    if freq > 0:
                        self._bass.append(_BassVoice.create(freq, self.sample_rate))
                elif kind == "kick":
                    self._drums.append(_KickVoice.create(self.sample_rate))
                elif kind == "snare":
                    self._drums.append(_SnareVoice.create(self.sample_rate))
        except Exception:
            pass

        # Sustained chord synthesis (for "hold fist to hold chord")
        if KEYS_ENABLED and KEYS_VOLUME > 0 and self._sustain:
            t_idx = np.arange(frames, dtype=np.float64)
            keep_s: List[_SustainChordVoice] = []
            # advance vibrato LFO phase for this block
            vib_inc = 2.0 * np.pi * float(KEYS_VIBRATO_HZ) / float(self.sample_rate)
            vib_phase0 = self._vib_phase
            self._vib_phase = float((self._vib_phase + vib_inc * float(frames)) % (2.0 * np.pi))
            vib = np.sin(vib_phase0 + vib_inc * t_idx)
            vib_ratio = 2.0 ** ((live_vib_cents / 1200.0) * vib)

            for v in self._sustain:
                y = v.render(t_idx, vib_ratio=vib_ratio)
                if y is not None:
                    yy = (y.astype(np.float32) * float(KEYS_VOLUME)).reshape(-1, 1)
                    out[:, :1] += yy
                    if self.channels > 1:
                        out[:, 1:2] += yy
                if not v.done:
                    keep_s.append(v)
            self._sustain = keep_s

        # Chord synthesis
        if KEYS_ENABLED and KEYS_VOLUME > 0:
            try:
                pass
            except Exception:
                pass

            if self._voices:
                t_idx = np.arange(frames, dtype=np.float64)
                keep: List[_ChordVoice] = []
                for v in self._voices:
                    y = v.render(t_idx)
                    if y is not None:
                        yy = (y.astype(np.float32) * float(KEYS_VOLUME)).reshape(-1, 1)
                        out[:, :1] += yy
                        if self.channels > 1:
                            out[:, 1:2] += yy
                    if not v.done:
                        keep.append(v)
                self._voices = keep

        # Bass synthesis
        if BASS_ENABLED and BASS_VOLUME > 0 and self._bass:
            t_idx = np.arange(frames, dtype=np.float64)
            keep_b: List[_BassVoice] = []
            for v in self._bass:
                y = v.render(t_idx)
                if y is not None:
                    yy = (y.astype(np.float32) * float(BASS_VOLUME)).reshape(-1, 1)
                    out[:, :1] += yy
                    if self.channels > 1:
                        out[:, 1:2] += yy
                if not v.done:
                    keep_b.append(v)
            self._bass = keep_b

        # Drum events + synthesis (independent of KEYS)
        if KIT_ENABLED and (KIT_KICK_VOLUME > 0 or KIT_SNARE_VOLUME > 0):
            if self._drums:
                t_idx = np.arange(frames, dtype=np.float64)
                keep_d: List[_DrumVoice] = []
                for d in self._drums:
                    y = d.render(t_idx)
                    if y is not None:
                        yy = y.astype(np.float32).reshape(-1, 1)
                        out[:, :1] += yy
                        if self.channels > 1:
                            out[:, 1:2] += yy
                    if not d.done:
                        keep_d.append(d)
                self._drums = keep_d

        outdata[:] = out
        with self._lock:
            self._sample_index += frames

    def beat_in_bar(self) -> int:
        # Based on sample counter: 1..4 looping
        b = self.beats_elapsed()
        return (b % 4) + 1

    def beats_elapsed(self) -> int:
        if self._beat_samples <= 0:
            return 0
        return int(self._sample_index // self._beat_samples)

    def bar_in_loop(self, bars_per_loop: int = 8) -> int:
        # 1..bars_per_loop looping
        b = self.beats_elapsed()
        return ((b // 4) % max(1, bars_per_loop)) + 1

    def play_chord(self, freqs_hz: List[float]) -> None:
        """Schedule a chord (list of frequencies) to play ASAP in the audio callback."""
        self._events.put(("chord", list(freqs_hz)))

    def start_hold_chord(self, freqs_hz: List[float]) -> None:
        self._events.put(("chord_on", list(freqs_hz)))

    def stop_hold_chord(self) -> None:
        self._events.put(("chord_off", None))

    def set_live_vibrato_cents(self, cents: float) -> None:
        with self._lock:
            self._live_vibrato_cents = float(max(0.0, cents))

    def play_bass(self, freq_hz: float) -> None:
        self._events.put(("bass", float(freq_hz)))

    def play_kick(self) -> None:
        self._events.put(("kick", None))

    def play_snare(self) -> None:
        self._events.put(("snare", None))

    def set_playback_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._playback_enabled = bool(enabled)

    def set_loop_layers(self, layers: List[List[tuple[int, str, object]]]) -> None:
        # Copy to avoid mutation races.
        safe: List[List[tuple[int, str, object]]] = []
        for layer in layers:
            safe.append([(int(o), str(k), p) for (o, k, p) in layer])
        with self._lock:
            self._loop_layers = safe

    def current_sample(self) -> int:
        with self._lock:
            return int(self._sample_index)


@dataclass
class _ChordVoice:
    freqs: np.ndarray  # (n,)
    phases: np.ndarray  # (n,)
    pos: int
    length: int
    attack: int
    decay: int
    sample_rate: int
    done: bool = False

    @staticmethod
    def from_freqs(freqs: np.ndarray, sample_rate: int) -> "_ChordVoice":
        length = max(1, int(sample_rate * (KEYS_DURATION_MS / 1000.0)))
        attack = max(1, int(sample_rate * (KEYS_ATTACK_MS / 1000.0)))
        decay = max(1, int(sample_rate * (KEYS_DECAY_MS / 1000.0)))
        phases = np.random.rand(freqs.size).astype(np.float64) * (2.0 * np.pi)
        return _ChordVoice(freqs=freqs, phases=phases, pos=0, length=length, attack=attack, decay=decay, sample_rate=sample_rate)

    def render(self, t_idx: np.ndarray) -> Optional[np.ndarray]:
        if self.done:
            return None
        remaining = self.length - self.pos
        if remaining <= 0:
            self.done = True
            return None

        n = int(min(t_idx.size, remaining))
        tt = t_idx[:n]
        inc = (2.0 * np.pi * self.freqs) / float(self.sample_rate)
        phases = self.phases[:, None] + inc[:, None] * tt[None, :]
        wave = np.sin(phases).sum(axis=0) / max(1.0, float(self.freqs.size))

        s = (self.pos + tt).astype(np.float64)
        env_attack = np.clip(s / float(self.attack), 0.0, 1.0)
        env_decay = np.exp(-s / float(self.decay))
        env = env_attack * env_decay

        # update phase + pos
        self.phases = (self.phases + inc * float(n)) % (2.0 * np.pi)
        self.pos += n
        if self.pos >= self.length:
            self.done = True
        # pad if block longer than remaining
        if n < t_idx.size:
            out = np.zeros((t_idx.size,), dtype=np.float64)
            out[:n] = wave * env
            return out
        return wave * env


@dataclass
class _SustainChordVoice:
    freqs: np.ndarray
    phases: np.ndarray
    pos: int
    attack: int
    decay: int
    release_samps: int
    sample_rate: int
    releasing: bool = False
    release_pos: int = 0
    done: bool = False

    @staticmethod
    def from_freqs(freqs: np.ndarray, sample_rate: int) -> "_SustainChordVoice":
        attack = max(1, int(sample_rate * (KEYS_ATTACK_MS / 1000.0)))
        decay = max(1, int(sample_rate * (KEYS_DECAY_MS / 1000.0)))
        release_samps = max(1, int(sample_rate * 0.12))
        phases = np.random.rand(freqs.size).astype(np.float64) * (2.0 * np.pi)
        return _SustainChordVoice(freqs=freqs, phases=phases, pos=0, attack=attack, decay=decay, release_samps=release_samps, sample_rate=sample_rate)

    def release(self) -> None:
        if not self.releasing:
            self.releasing = True
            self.release_pos = 0

    def render(self, t_idx: np.ndarray, *, vib_ratio: np.ndarray) -> Optional[np.ndarray]:
        if self.done:
            return None

        n = int(t_idx.size)
        # per-sample phase inc with vibrato
        inc = (2.0 * np.pi * self.freqs) / float(self.sample_rate)
        # integrate phase with varying step (approx)
        # Use base phase + cumulative sum of (inc * vib_ratio)
        step = (inc[:, None] * vib_ratio[None, :])
        phases = self.phases[:, None] + np.cumsum(step, axis=1)
        wave = np.sin(phases).sum(axis=0) / max(1.0, float(self.freqs.size))

        s = (self.pos + t_idx).astype(np.float64)
        env_attack = np.clip(s / float(self.attack), 0.0, 1.0)
        env_decay = np.exp(-s / float(self.decay))
        env = env_attack * env_decay

        if self.releasing:
            r = np.clip(1.0 - (self.release_pos + t_idx) / float(self.release_samps), 0.0, 1.0)
            env = env * r
            self.release_pos += n
            if self.release_pos >= self.release_samps:
                self.done = True

        # update phases (advance by last column)
        self.phases = (phases[:, -1] % (2.0 * np.pi)).astype(np.float64)
        self.pos += n
        return wave * env


@dataclass
class _DrumVoice:
    pos: int
    length: int
    sample_rate: int
    done: bool = field(default=False, init=False)

    def render(self, t_idx: np.ndarray) -> Optional[np.ndarray]:
        raise NotImplementedError


@dataclass
class _KickVoice(_DrumVoice):
    @staticmethod
    def create(sample_rate: int) -> "_KickVoice":
        # Deeper kick: low pitch drop + longer body + tiny click.
        length = max(1, int(sample_rate * 0.28))
        return _KickVoice(pos=0, length=length, sample_rate=sample_rate)

    def render(self, t_idx: np.ndarray) -> Optional[np.ndarray]:
        if self.done:
            return None
        remaining = self.length - self.pos
        if remaining <= 0:
            self.done = True
            return None
        n = int(min(t_idx.size, remaining))
        tt = (self.pos + t_idx[:n]) / float(self.sample_rate)
        # Exponential pitch drop (deeper)
        f0, f1 = 85.0, 34.0
        f = f1 + (f0 - f1) * np.exp(-tt / 0.05)
        phase = 2.0 * np.pi * np.cumsum(f) / float(self.sample_rate)
        env = np.exp(-tt / 0.24)
        attack = np.clip(tt / 0.004, 0.0, 1.0)
        body = np.sin(phase) * env * attack
        sub = np.sin(phase * 0.5) * env * 0.35
        # tiny click at onset
        click = (np.random.randn(n) * np.exp(-tt / 0.005) * 0.06)
        y = body + sub + click
        # a touch of saturation for punch
        y = np.tanh(y * float(KICK_DRIVE))
        y = y * float(KIT_KICK_VOLUME)
        self.pos += n
        if self.pos >= self.length:
            self.done = True
        if n < t_idx.size:
            out = np.zeros((t_idx.size,), dtype=np.float64)
            out[:n] = y
            return out
        return y


@dataclass
class _SnareVoice(_DrumVoice):
    @staticmethod
    def create(sample_rate: int) -> "_SnareVoice":
        length = max(1, int(sample_rate * 0.22))
        return _SnareVoice(pos=0, length=length, sample_rate=sample_rate)

    def render(self, t_idx: np.ndarray) -> Optional[np.ndarray]:
        if self.done:
            return None
        remaining = self.length - self.pos
        if remaining <= 0:
            self.done = True
            return None
        n = int(min(t_idx.size, remaining))
        tt = (self.pos + t_idx[:n]) / float(self.sample_rate)
        env = np.exp(-tt / 0.14)
        noise = np.random.randn(n).astype(np.float64)

        # Band-pass-ish noise (HP then LP) so it sounds like a snare, not a hat.
        sr = float(self.sample_rate)
        hp_a = np.exp(-2.0 * np.pi * float(SNARE_HP_HZ) / sr)
        lp_a = np.exp(-2.0 * np.pi * float(SNARE_LP_HZ) / sr)
        # one-pole HP via subtracting one-pole LP
        lp = np.empty_like(noise)
        ylp = 0.0
        for i in range(n):
            ylp = (hp_a * ylp) + ((1.0 - hp_a) * noise[i])
            lp[i] = ylp
        hp = noise - lp
        # then one-pole LP
        bp = np.empty_like(hp)
        ybp = 0.0
        for i in range(n):
            ybp = (lp_a * ybp) + ((1.0 - lp_a) * hp[i])
            bp[i] = ybp

        tone = np.sin(2.0 * np.pi * float(SNARE_TONE_HZ) * tt)
        y = (0.82 * bp + 0.18 * tone) * env
        y = np.tanh(y * 1.2) * float(KIT_SNARE_VOLUME)
        self.pos += n
        if self.pos >= self.length:
            self.done = True
        if n < t_idx.size:
            out = np.zeros((t_idx.size,), dtype=np.float64)
            out[:n] = y
            return out
        return y


@dataclass
class _BassVoice:
    freq: float
    phase: float
    pos: int
    length: int
    attack: int
    decay: int
    sample_rate: int
    done: bool = False

    @staticmethod
    def create(freq: float, sample_rate: int) -> "_BassVoice":
        length = max(1, int(sample_rate * (BASS_DURATION_MS / 1000.0)))
        attack = max(1, int(sample_rate * (BASS_ATTACK_MS / 1000.0)))
        decay = max(1, int(sample_rate * (BASS_DECAY_MS / 1000.0)))
        return _BassVoice(freq=float(freq), phase=float(np.random.rand() * 2.0 * np.pi), pos=0, length=length, attack=attack, decay=decay, sample_rate=sample_rate)

    def render(self, t_idx: np.ndarray) -> Optional[np.ndarray]:
        if self.done:
            return None
        remaining = self.length - self.pos
        if remaining <= 0:
            self.done = True
            return None

        n = int(min(t_idx.size, remaining))
        tt = t_idx[:n]
        inc = (2.0 * np.pi * float(self.freq)) / float(self.sample_rate)
        phase = self.phase + inc * tt

        # More "instrument-like" bass: band-limited-ish saw (harmonic sum) + sub, then low-pass + drive.
        H = int(max(1, min(32, int(BASS_HARMONICS))))
        saw = np.zeros(n, dtype=np.float64)
        for k in range(1, H + 1):
            saw += np.sin(phase * float(k)) / float(k)
        saw *= (2.0 / np.pi)
        sub = np.sin(phase * 0.5)
        wave = (1.0 - float(BASS_SUB_MIX)) * saw + float(BASS_SUB_MIX) * sub

        s = (self.pos + tt).astype(np.float64)
        env_attack = np.clip(s / float(self.attack), 0.0, 1.0)
        env_decay = np.exp(-s / float(self.decay))
        env = env_attack * env_decay

        # update
        self.phase = float((self.phase + inc * float(n)) % (2.0 * np.pi))
        self.pos += n
        if self.pos >= self.length:
            self.done = True

        y = wave * env
        # one-pole low-pass for warmth
        a = np.exp(-2.0 * np.pi * float(BASS_LPF_HZ) / float(self.sample_rate))
        ylp = 0.0
        for i in range(n):
            ylp = (a * ylp) + ((1.0 - a) * y[i])
            y[i] = ylp
        # gentle saturation
        y = np.tanh(y * float(BASS_DRIVE))
        if n < t_idx.size:
            out = np.zeros((t_idx.size,), dtype=np.float64)
            out[:n] = y
            return out
        return y

ASCII_LOGO = [
                                                              
    r"                     _                      _   _             ",
    r" _ __ ___  _   _ ___(_) ___ _ __ ___   ___ | |_(_) ___  _ __  ",
    r"| '_ ` _ \| | | / __| |/ __| '_ ` _ \ / _ \| __| |/ _ \| '_ \ ",
    r"| | | | | | |_| \__ \ | (__| | | | | | (_) | |_| | (_) | | | |",
    r"|_| |_| |_|\__,_|___/_|\___|_| |_| |_|\___/ \__|_|\___/|_| |_|",
]


def draw_ascii_banner(canvas_bgr: np.ndarray, title_lines=ASCII_LOGO) -> None:
    h, w = canvas_bgr.shape[:2]
    canvas_bgr[:] = (18, 18, 18)

    green = (40, 255, 120)
    shadow = (0, 40, 0)

    font = cv2.FONT_HERSHEY_PLAIN
    scale = 1.15
    thickness = 2

    char_w = cv2.getTextSize("M", font, scale, thickness)[0][0]
    line_h = cv2.getTextSize("A", font, scale, thickness)[0][1] + 8
    block_h = line_h * len(title_lines)
    y0 = max(8, (h - block_h) // 2)
    x0 = 14

    for i, line in enumerate(title_lines):
        y = y0 + (i + 1) * line_h
        for j, ch in enumerate(line):
            x = x0 + j * char_w
            cv2.putText(canvas_bgr, ch, (x + 3, y + 3), font, scale, shadow, thickness + 4, cv2.LINE_AA)
            cv2.putText(canvas_bgr, ch, (x + 2, y + 2), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(canvas_bgr, ch, (x, y), font, scale, green, thickness, cv2.LINE_AA)

    for y in range(0, h, 4):
        canvas_bgr[y : y + 1, :] = (12, 12, 12)
    cv2.line(canvas_bgr, (0, h - 1), (w - 1, h - 1), (60, 60, 60), 1)


def _mirror_hands(hands: List[HandPosition], *, width_px: int) -> List[HandPosition]:
    out: List[HandPosition] = []
    for hand in hands:
        lms: List[HandLandmark] = []
        for lm in hand.landmarks:
            x_px = int(width_px - 1 - lm.x_px)
            lms.append(
                HandLandmark(
                    idx=lm.idx,
                    x_norm=1.0 - lm.x_norm,
                    y_norm=lm.y_norm,
                    z_norm=lm.z_norm,
                    x_px=x_px,
                    y_px=lm.y_px,
                )
            )

        x0, y0, x1, y1 = hand.bbox_px
        bbox = (int(width_px - 1 - x1), y0, int(width_px - 1 - x0), y1)
        cx, cy = hand.center_px
        center = (int(width_px - 1 - cx), cy)
        tips = {k: (int(width_px - 1 - v[0]), v[1]) for k, v in hand.fingertips_px.items()}

        out.append(
            HandPosition(
                handedness_label=hand.handedness_label,
                handedness_score=hand.handedness_score,
                landmarks=lms,
                bbox_px=bbox,
                center_px=center,
                fingertips_px=tips,
            )
        )
    return out


def _right_hand(hands: List[HandPosition]) -> Optional[HandPosition]:
    # Prefer explicit Right hand; fall back to first hand.
    for h in hands:
        if (h.handedness_label or "").lower() == "right":
            return h
    return hands[0] if hands else None


def _left_hand(hands: List[HandPosition]) -> Optional[HandPosition]:
    for h in hands:
        if (h.handedness_label or "").lower() == "left":
            return h
    # no fallback; left-hand control is optional
    return None


def _right_hand_strict(hands: List[HandPosition]) -> Optional[HandPosition]:
    """Return the hand labeled Right only (no fallback)."""
    for h in hands:
        if (h.handedness_label or "").lower() == "right":
            return h
    return None


def _midi_to_hz(midi: int) -> float:
    return 440.0 * (2.0 ** ((float(midi) - 69.0) / 12.0))


def _degree_chords(mode: str) -> List[Tuple[str, str, List[int]]]:
    """
    Returns 7 diatonic triads as (roman, label, midi_notes).

    - Major => C major (C D E F G A B)
    - Minor => A minor with V as E major (as requested)
    """

    m = (mode or "").lower()

    if m == "minor":
        # A minor: i, ii°, III, iv, V (E major), VI, VII
        roots = [57, 59, 60, 62, 64, 65, 67]  # A3..G4
        qualities = ["min", "dim", "maj", "min", "maj", "maj", "maj"]
        romans = ["i", "ii°", "III", "iv", "V", "VI", "VII"]
        labels = ["Am", "Bdim", "C", "Dm", "E", "F", "G"]
    else:
        # C major: I, ii, iii, IV, V, vi, vii°
        roots = [60, 62, 64, 65, 67, 69, 59]  # keep Bdim lower (B3)
        qualities = ["maj", "min", "min", "maj", "maj", "min", "dim"]
        romans = ["I", "ii", "iii", "IV", "V", "vi", "vii°"]
        labels = ["C", "Dm", "Em", "F", "G", "Am", "Bdim"]

    out: List[Tuple[str, str, List[int]]] = []
    for root, q, rn, lb in zip(roots, qualities, romans, labels):
        if q == "maj":
            notes = [root, root + 4, root + 7]
        elif q == "min":
            notes = [root, root + 3, root + 7]
        else:  # dim
            notes = [root, root + 3, root + 6]
        out.append((rn, lb, notes))
    return out


def _draw_chord_ruler(frame, rect, chords: List[Tuple[str, str, List[int]]], selected_idx: int) -> None:
    x0, y0, x1, y1 = rect
    _draw_rect_alpha(frame, rect, (18, 18, 18), alpha=0.55)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (80, 80, 80), 2)
    cv2.putText(frame, "CHORDS", (x0 + 10, y0 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    inner_y0 = y0 + 40
    inner_h = max(1, (y1 - inner_y0 - 10))
    seg_h = inner_h / max(1, len(chords))

    for i, (rn, lb, _notes) in enumerate(chords):
        yy0 = int(inner_y0 + i * seg_h)
        yy1 = int(inner_y0 + (i + 1) * seg_h)
        if i == selected_idx:
            _draw_rect_alpha(frame, (x0 + 6, yy0 + 2, x1 - 6, yy1 - 2), (40, 255, 120), alpha=0.72)
            col = (10, 10, 10)
        else:
            col = (235, 235, 235)

        cv2.putText(frame, rn, (x0 + 10, yy0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
        cv2.putText(frame, lb, (x0 + 58, yy0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2, cv2.LINE_AA)


def _hand_lr_label(hand: HandPosition) -> str:
    lab = (hand.handedness_label or "").lower()
    if lab == "left":
        return "L"
    if lab == "right":
        return "R"
    return "?"


def _hand_openness_score(hand: HandPosition) -> float:
    """
    0.0 = closed fist-ish, 1.0 = open-ish.
    Uses fingertip-to-MCP distances normalized by bbox diagonal.
    """

    lms = hand.landmarks
    if len(lms) < 21:
        return 1.0
    # Use a stable scale: palm length (wrist -> middle MCP). This is much less sensitive than bbox size.
    wrist = (lms[0].x_px, lms[0].y_px)
    palm = (lms[9].x_px, lms[9].y_px)
    palm_len = max(1.0, _dist(wrist, palm))

    pairs = [
        (8, 5),  # index tip -> index mcp
        (12, 9),  # middle
        (16, 13),  # ring
        (20, 17),  # pinky
        (4, 2),  # thumb tip -> thumb mcp-ish
    ]
    dsum = 0.0
    for a, b in pairs:
        dx = float(lms[a].x_px - lms[b].x_px)
        dy = float(lms[a].y_px - lms[b].y_px)
        dsum += (dx * dx + dy * dy) ** 0.5
    davg = dsum / len(pairs)
    norm = davg / palm_len

    # Map a rough range into 0..1 for easier thresholds.
    # With palm normalization, typical ranges are closer to:
    # - closed: ~0.35-0.65
    # - open:   ~0.80-1.40 (varies by angle)
    return float(np.clip((norm - 0.55) / (1.25 - 0.55), 0.0, 1.0))


def _fist_score(*, openness: float, tip_norm: float) -> float:
    """
    Convert raw measurements into a 0..1 closedness score (higher = more closed).

    NOTE: openness can be imperfect depending on angle, so we weight fingertip proximity more.
    """

    closed_from_open = 1.0 - float(np.clip(openness, 0.0, 1.0))
    # tip_norm: smaller means tips closer to palm (more closed)
    closed_from_tip = 1.0 - float(np.clip((tip_norm - 0.45) / (1.10 - 0.45), 0.0, 1.0))
    return float(np.clip(0.20 * closed_from_open + 0.80 * closed_from_tip, 0.0, 1.0))


@dataclass
class DirectionGate:
    """
    Stabilizes direction gestures (to reduce finicky behavior):
    - requires the same direction for N consecutive frames
    - applies a cooldown between triggers
    """

    last_dir: Optional[str] = None
    stable_count: int = 0
    last_trigger_t: float = 0.0
    armed: bool = True

    def update_and_trigger(self, direction: Optional[str], *, stable_frames: int, cooldown_s: float) -> Optional[str]:
        now = time.time()
        if direction is None:
            self.last_dir = None
            self.stable_count = 0
            self.armed = True
            return None

        if direction == self.last_dir:
            self.stable_count += 1
        else:
            self.last_dir = direction
            self.stable_count = 1
            self.armed = True

        if self.armed and self.stable_count >= stable_frames and (now - self.last_trigger_t) >= cooldown_s:
            self.last_trigger_t = now
            self.stable_count = 0
            self.armed = False
            return direction

        return None


@dataclass
class BoolEdgeGate:
    """Rising-edge trigger for a boolean condition, with cooldown."""

    prev: bool = False
    last_trigger_t: float = 0.0

    def rising_edge(self, curr: bool, *, cooldown_s: float) -> bool:
        now = time.time()
        fired = False
        if (not self.prev) and curr and (now - self.last_trigger_t) >= cooldown_s:
            self.last_trigger_t = now
            fired = True
        self.prev = curr
        return fired


@dataclass
class ClenchGate:
    """
    Hand clench gate with hysteresis:
    - becomes 'closed' when value <= close_th
    - becomes 'open' when value >= open_th
    - triggers ONLY on open->closed transition (with cooldown)
    """

    is_closed: bool = False
    last_fire_t: float = 0.0

    def update_and_fire(self, value: float, *, close_th: float, open_th: float, cooldown_s: float) -> bool:
        now = time.time()

        # reopen
        if self.is_closed:
            if value >= open_th:
                self.is_closed = False
            return False

        # open -> closed edge
        if value <= close_th:
            self.is_closed = True
            if (now - self.last_fire_t) >= cooldown_s:
                self.last_fire_t = now
                return True
        return False


@dataclass
class HoldGate:
    """
    Fires when `cond` has been true continuously for `hold_s`,
    with a cooldown to prevent repeats while still holding.
    """

    hold_start_t: Optional[float] = None
    last_fire_t: float = 0.0

    def reset(self) -> None:
        self.hold_start_t = None

    def update(self, cond: bool, *, hold_s: float, cooldown_s: float) -> bool:
        now = time.time()
        if not cond:
            self.hold_start_t = None
            return False

        if self.hold_start_t is None:
            self.hold_start_t = now
            return False

        if (now - self.hold_start_t) >= hold_s and (now - self.last_fire_t) >= cooldown_s:
            self.last_fire_t = now
            # keep hold_start_t so holding continues, but cooldown prevents spam
            return True

        return False


HAND_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    r: float


def _update_and_draw_particles(frame, particles: List[Particle], dt: float) -> None:
    if not particles:
        return
    out: List[Particle] = []
    layer = np.zeros_like(frame)
    for p in particles:
        p.life -= dt
        if p.life <= 0:
            continue
        p.x += p.vx * dt
        p.y += p.vy * dt
        # Flame-ish drift: keep rising, slight damping, add a tiny flicker.
        p.vx = p.vx * 0.97 + float(np.random.randn() * 6.0)
        p.vy = p.vy * 0.99 - 4.0
        out.append(p)

        # Square particles, pure white.
        s = int(max(2.0, p.r))
        x0 = int(p.x - s)
        y0 = int(p.y - s)
        x1 = int(p.x + s)
        y1 = int(p.y + s)
        cv2.rectangle(layer, (x0, y0), (x1, y1), (255, 255, 255), -1)

    # Additive-ish blend for a brighter "bloom" look.
    cv2.add(frame, layer, dst=frame)
    particles[:] = out


def _spawn_hand_particles(particles: List[Particle], hand: HandPosition, n: int = 10) -> None:
    if len(hand.landmarks) < 21:
        return
    emit_idxs = [0, 4, 8, 12, 16, 20, 9]  # wrist + fingertips + palm-ish
    for _ in range(n):
        idx = emit_idxs[int(np.random.randint(0, len(emit_idxs)))]
        lm = hand.landmarks[idx]
        jitter = np.random.randn(2) * 5.0
        # Strong upward bias, small spread sideways -> flame plume.
        vx = float(np.random.randn() * 28.0)
        vy = float(-120.0 - np.random.rand() * 120.0 + np.random.randn() * 14.0)
        particles.append(
            Particle(
                x=float(lm.x_px + jitter[0]),
                y=float(lm.y_px + jitter[1]),
                vx=vx,
                vy=vy,
                life=float(0.40 + np.random.rand() * 0.55),
                r=float(3.0 + np.random.rand() * 4.5),
            )
        )


def _draw_hand_bbox_alpha(frame, hand: HandPosition, *, alpha: float = 0.16) -> None:
    x0, y0, x1, y1 = hand.bbox_px
    _draw_rect_alpha(frame, (x0, y0, x1, y1), (255, 255, 255), alpha=alpha)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 255), 1, cv2.LINE_AA)


def _draw_retro_hand(frame, hand: HandPosition) -> None:
    for a, b in HAND_CONNECTIONS:
        if a >= len(hand.landmarks) or b >= len(hand.landmarks):
            continue
        p0 = (hand.landmarks[a].x_px, hand.landmarks[a].y_px)
        p1 = (hand.landmarks[b].x_px, hand.landmarks[b].y_px)
        cv2.line(frame, p0, p1, (30, 30, 30), 4, cv2.LINE_AA)
        cv2.line(frame, p0, p1, (255, 255, 255), 2, cv2.LINE_AA)

    for lm in hand.landmarks:
        cv2.circle(frame, (lm.x_px, lm.y_px), 5, (30, 30, 30), -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (lm.x_px, lm.y_px), 3, (255, 255, 255), -1, lineType=cv2.LINE_AA)


def _draw_hand_label(frame, hand: HandPosition) -> None:
    x0, y0, x1, y1 = hand.bbox_px
    label = _hand_lr_label(hand)
    org = (x0 + 8, max(28, y0 - 10))
    cv2.putText(frame, label, (org[0] + 2, org[1] + 2), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, label, org, cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)

    # Closedness strength label (for tuning)
    try:
        if len(hand.landmarks) < 21:
            return
        openness = _hand_openness_score(hand)
        wrist = (hand.landmarks[0].x_px, hand.landmarks[0].y_px)
        palm = (hand.landmarks[9].x_px, hand.landmarks[9].y_px)
        palm_len = max(1.0, _dist(wrist, palm))
        cx, cy = hand.center_px
        tip_idxs = [4, 8, 12, 16, 20]
        dsum = 0.0
        for i in tip_idxs:
            dsum += _dist((hand.landmarks[i].x_px, hand.landmarks[i].y_px), (cx, cy))
        tip_norm = (dsum / len(tip_idxs)) / palm_len
        score = _fist_score(openness=openness, tip_norm=tip_norm)
        txt = f"C:{score:0.2f}  o:{openness:0.2f}  t:{tip_norm:0.2f}"
        cv2.putText(frame, txt, (org[0] + 26, org[1] + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, txt, (org[0] + 24, org[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    except Exception:
        pass


def _point_direction(hand: HandPosition) -> Optional[str]:
    """
    Infer a simple pointing direction from index finger vector.
    Uses index MCP (5) -> index TIP (8).
    """
    lms = hand.landmarks
    if len(lms) < 9:
        return None

    x0, y0 = float(lms[5].x_px), float(lms[5].y_px)
    x1, y1 = float(lms[8].x_px), float(lms[8].y_px)
    vx, vy = x1 - x0, y1 - y0
    mag = (vx * vx + vy * vy) ** 0.5
    if mag < 35.0:
        return None
    vx /= mag
    vy /= mag

    # Cheap "pointing" heuristic: index extension should be >= middle extension.
    if len(lms) >= 13:
        mx0, my0 = float(lms[9].x_px), float(lms[9].y_px)
        mx1, my1 = float(lms[12].x_px), float(lms[12].y_px)
        mlen = ((mx1 - mx0) ** 2 + (my1 - my0) ** 2) ** 0.5
        if mag < mlen * 0.95:
            return None

    if vy < -0.78 and abs(vx) < 0.55:
        return "up"
    if vy > 0.78 and abs(vx) < 0.55:
        return "down"
    if vx < -0.78 and abs(vy) < 0.55:
        return "left"
    if vx > 0.78 and abs(vy) < 0.55:
        return "right"
    return None


def _dist(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    dx = float(a[0] - b[0])
    dy = float(a[1] - b[1])
    return (dx * dx + dy * dy) ** 0.5


def _all_fingers_closed(hand: HandPosition) -> bool:
    """
    Stricter "fist" check: ALL fingers must be folded (thumb + 4 fingers).
    Uses wrist-distance comparison to detect folding robustly.
    """

    lms = hand.landmarks
    if len(lms) < 21:
        return False

    wrist = (lms[0].x_px, lms[0].y_px)
    x0, y0, x1, y1 = hand.bbox_px
    diag = max(1.0, float(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5))

    def folded(tip: int, mcp: int, slack: float) -> bool:
        d_tip = _dist((lms[tip].x_px, lms[tip].y_px), wrist)
        d_mcp = _dist((lms[mcp].x_px, lms[mcp].y_px), wrist)
        # tip should not extend beyond MCP by more than small slack
        return d_tip <= d_mcp + slack * diag

    # Tuned slack values: more forgiving in practice (camera angle varies a lot).
    return (
        folded(4, 2, 0.06)  # thumb tip vs thumb MCP-ish
        and folded(8, 5, 0.05)  # index
        and folded(12, 9, 0.05)  # middle
        and folded(16, 13, 0.05)  # ring
        and folded(20, 17, 0.05)  # pinky
    )


def _is_fist(hand: HandPosition) -> bool:
    """
    Practical fist detector (less finicky than per-finger checks):
    - low overall openness
    - fingertips close to palm center
    """

    if len(hand.landmarks) < 21:
        return False

    openness = _hand_openness_score(hand)
    wrist = (hand.landmarks[0].x_px, hand.landmarks[0].y_px)
    palm = (hand.landmarks[9].x_px, hand.landmarks[9].y_px)
    palm_len = max(1.0, _dist(wrist, palm))
    cx, cy = hand.center_px

    tip_idxs = [4, 8, 12, 16, 20]
    dsum = 0.0
    for i in tip_idxs:
        dsum += _dist((hand.landmarks[i].x_px, hand.landmarks[i].y_px), (cx, cy))
    avg_tip = dsum / len(tip_idxs)
    tip_norm = avg_tip / palm_len

    return _fist_score(openness=openness, tip_norm=tip_norm) >= FIST_SCORE_THRESHOLD


def _two_fists_closed(hands: List[HandPosition]) -> bool:
    """
    Require two fists (both hands).
    Prefers explicit Left+Right labels if present; otherwise requires any 2 closed hands.
    """

    if len(hands) < 2:
        return False

    left = None
    right = None
    for h in hands:
        lab = (h.handedness_label or "").lower()
        if lab == "left":
            left = h
        elif lab == "right":
            right = h

    if left is not None and right is not None:
        return _is_fist(left) and _is_fist(right)

    # Otherwise: compute scores for all hands and require the top 2 to clear the threshold.
    scores: List[float] = []
    for h in hands:
        if len(h.landmarks) < 21:
            continue
        o = _hand_openness_score(h)
        wrist = (h.landmarks[0].x_px, h.landmarks[0].y_px)
        palm = (h.landmarks[9].x_px, h.landmarks[9].y_px)
        palm_len = max(1.0, _dist(wrist, palm))
        cx, cy = h.center_px
        tip_idxs = [4, 8, 12, 16, 20]
        dsum = 0.0
        for i in tip_idxs:
            dsum += _dist((h.landmarks[i].x_px, h.landmarks[i].y_px), (cx, cy))
        tip_norm = (dsum / len(tip_idxs)) / palm_len
        scores.append(_fist_score(openness=o, tip_norm=tip_norm))

    scores.sort(reverse=True)
    return len(scores) >= 2 and scores[0] >= FIST_SCORE_THRESHOLD and scores[1] >= FIST_SCORE_THRESHOLD


def _draw_rect_alpha(frame, rect, color_bgr, alpha: float) -> None:
    """Alpha-blend a solid rect on top of the frame."""
    x0, y0, x1, y1 = rect
    x0 = max(0, int(x0))
    y0 = max(0, int(y0))
    x1 = min(frame.shape[1], int(x1))
    y1 = min(frame.shape[0], int(y1))
    if x1 <= x0 or y1 <= y0:
        return

    roi = frame[y0:y1, x0:x1]
    overlay = np.empty_like(roi)
    overlay[:, :] = color_bgr
    cv2.addWeighted(overlay, float(alpha), roi, float(1.0 - alpha), 0.0, dst=roi)


def _blend_image_alpha(dst_bgr, src_bgr, x: int, y: int, alpha: float) -> None:
    """Alpha-blend `src_bgr` onto `dst_bgr` at top-left (x,y)."""
    if alpha <= 0.0:
        return
    h, w = src_bgr.shape[:2]
    H, W = dst_bgr.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(W, x + w)
    y1 = min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    sx0 = x0 - x
    sy0 = y0 - y
    sx1 = sx0 + (x1 - x0)
    sy1 = sy0 + (y1 - y0)
    roi = dst_bgr[y0:y1, x0:x1]
    src = src_bgr[sy0:sy1, sx0:sx1]
    cv2.addWeighted(src, float(alpha), roi, float(1.0 - alpha), 0.0, dst=roi)


def _draw_panel(frame, rect, title: str, active: bool, selected_idx: int, items: List[str]) -> None:
    x0, y0, x1, y1 = rect
    bg = (28, 28, 28)
    border = (40, 255, 120) if active else (80, 80, 80)
    _draw_rect_alpha(frame, rect, bg, alpha=0.62)
    cv2.rectangle(frame, (x0, y0), (x1, y1), border, 2)

    cv2.putText(frame, title, (x0 + 12, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    # Items list
    inner_top = y0 + 48
    row_h = 34
    for i, text in enumerate(items):
        yy = inner_top + i * row_h
        if yy + row_h > y1 - 10:
            break
        is_sel = i == selected_idx
        if is_sel:
            _draw_rect_alpha(frame, (x0 + 10, yy - 24, x1 - 10, yy + 8), (40, 255, 120), alpha=0.88)
            color = (10, 10, 10)
        else:
            color = (220, 220, 220)
        cv2.putText(frame, text, (x0 + 18, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


def _draw_info_panel(frame, rect, bpm: int, mode: str, extra_lines: Optional[List[Tuple[str, str]]] = None) -> None:
    x0, y0, x1, y1 = rect
    _draw_rect_alpha(frame, rect, (20, 20, 20), alpha=0.62)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (60, 60, 60), 2)
    cv2.putText(frame, "PROJECT", (x0 + 12, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    lines = [
        ("Time", "4/4"),
        ("Loop", "8 bars"),
        ("BPM", str(bpm)),
        ("Key", mode),
        ("Ctrl", "Right hand"),
    ]
    if extra_lines:
        lines.extend(extra_lines)
    yy = y0 + 62
    for k, v in lines:
        cv2.putText(frame, f"{k}:", (x0 + 12, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (170, 170, 170), 2, cv2.LINE_AA)
        cv2.putText(frame, f"{v}", (x0 + 90, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        yy += 28


def _draw_instrument_panel(frame, rect, instruments: List[str], selected_idx: int = 0) -> None:
    x0, y0, x1, y1 = rect
    _draw_rect_alpha(frame, rect, (18, 18, 18), alpha=0.62)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (80, 80, 80), 2)
    cv2.putText(frame, "INSTRUMENTS", (x0 + 12, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    yy = y0 + 68
    for i, name in enumerate(instruments):
        if i == selected_idx:
            _draw_rect_alpha(frame, (x0 + 10, yy - 24, x1 - 10, yy + 8), (40, 255, 120), alpha=0.75)
            col = (10, 10, 10)
        else:
            col = (230, 230, 230)
        cv2.putText(frame, name.upper(), (x0 + 16, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.85, col, 2, cv2.LINE_AA)
        yy += 42


def main() -> int:
    ap = argparse.ArgumentParser(description="MusicMotion title screen (hand-controlled).")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--banner-height", type=int, default=92)
    ap.add_argument("--no-mirror", action="store_true", help="Disable mirror (default is selfie mirror).")
    ap.add_argument("--tasks-model", default="models/hand_landmarker.task")
    args = ap.parse_args()

    # Camera
    if platform.system() == "Darwin":
        cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    window_name = "musicmotion - title"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    banner = np.zeros((args.banner_height, args.width, 3), dtype=np.uint8)
    draw_ascii_banner(banner)

    # State
    bpm_values = list(range(60, 181, 5))
    bpm_idx = bpm_values.index(120) if 120 in bpm_values else 0
    mode_items = ["Major", "Minor"]
    mode_idx = 0

    active_panel = "bpm"  # or "mode"
    dir_gate = DirectionGate()
    left_dir_gate = DirectionGate()
    left_fist_gate = BoolEdgeGate()
    right_fist_gate = BoolEdgeGate()
    kit_kick_gate = ClenchGate()
    kit_snare_gate = ClenchGate()
    keys_hold_gate = ClenchGate()
    keys_holding = False
    keys_baseline_y: Optional[int] = None
    last_t = time.time()
    fps = 0.0
    particles: List[Particle] = []
    proceed_hold = HoldGate()

    state: str = "welcome"  # welcome -> title -> countdown -> loop
    welcome_start_t: float = time.time()
    countdown_start_t: Optional[float] = None
    loop_start_t: Optional[float] = None
    instruments = ["keys", "kit", "bass", "lead"]
    instrument_idx = 0
    chord_slot_state = 3  # persistent chord selection (0..6); prevents jumping when R is missing
    locked_instrument_idx: Optional[int] = None
    loop_layers: List[List[tuple[int, str, object]]] = []
    current_layer: List[tuple[int, str, object]] = []

    metro = Metronome(sample_rate=48000, channels=2)
    metro_running = False
    prev_state: Optional[str] = None

    with HandPositionDetector(max_num_hands=2, tasks_model_path=args.tasks_model) as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            hands_raw = detector.detect(frame)

            # Mirrored display + coordinate mirroring (keeps handedness label stable).
            if not args.no_mirror:
                display = cv2.flip(frame, 1)
                fh, fw = frame.shape[:2]
                hands = _mirror_hands(hands_raw, width_px=fw)
            else:
                display = frame
                hands = hands_raw

            # Choose control hands.
            # - `rh` drives setup (may fall back)
            # - `rh_strict` drives chord selection ONLY (no fallback; prevents left-hand affecting pitch)
            rh = _right_hand(hands)
            rh_strict = _right_hand_strict(hands)
            lh = _left_hand(hands)

            # Update fps
            now = time.time()
            dt = max(1e-6, now - last_t)
            inst = 1.0 / dt
            fps = 0.85 * fps + 0.15 * inst if fps > 0 else inst
            last_t = now

            # Layout
            H, W = display.shape[:2]
            info_w = int(W * 0.24)
            pad = 12
            top = pad
            bottom = H - pad
            left_w = int((W - info_w - 3 * pad) * 0.55)
            right_w = (W - info_w - 3 * pad) - left_w

            bpm_rect = (pad, top, pad + left_w, bottom)
            mode_rect = (pad + left_w + pad, top, pad + left_w + pad + right_w, bottom)
            info_rect = (W - info_w - pad, top, W - pad, bottom)

            bpm = bpm_values[bpm_idx]
            mode = mode_items[mode_idx]

            # Start metronome ONLY for countdown/loop, and keep UI in sync with the audio clock.
            if METRONOME_ENABLED:
                if state in ("countdown", "loop"):
                    if not metro_running:
                        metro.set_bpm(bpm)
                        metro.start()
                        metro_running = True
                    else:
                        # bpm shouldn't change here, but keep it synced just in case
                        metro.set_bpm(bpm)
                else:
                    if metro_running:
                        metro.stop()
                        metro_running = False
            else:
                if metro_running:
                    metro.stop()
                    metro_running = False

            # Reset phase on entering countdown OR loop (so both start at bar1/beat1)
            if METRONOME_ENABLED and metro_running and (prev_state != state) and (state in ("countdown", "loop")):
                metro.set_bpm(bpm)
                metro.reset_phase()

            # Identify labeled hands early (used for proceed + debug).
            l_hand = next((h for h in hands if (h.handedness_label or "").lower() == "left"), None)
            r_hand = next((h for h in hands if (h.handedness_label or "").lower() == "right"), None)

            # Proceed condition: BOTH hands (L+R) present and both openness ~ 0.
            # This matches the UI debug values (L_open/R_open).
            two_fists = False
            if l_hand is not None and r_hand is not None:
                try:
                    l_open_val = float(_hand_openness_score(l_hand))
                    r_open_val = float(_hand_openness_score(r_hand))
                    two_fists = (l_open_val <= PROCEED_OPENNESS_MAX) and (r_open_val <= PROCEED_OPENNESS_MAX)
                except Exception:
                    two_fists = False

            did_proceed = proceed_hold.update(two_fists, hold_s=PROCEED_HOLD_S, cooldown_s=PROCEED_COOLDOWN_S)
            # Debug for fist detection
            l_fist = _is_fist(l_hand) if l_hand is not None else False
            r_fist = _is_fist(r_hand) if r_hand is not None else False
            l_open = f"{_hand_openness_score(l_hand):0.2f}" if l_hand is not None else "-"
            r_open = f"{_hand_openness_score(r_hand):0.2f}" if r_hand is not None else "-"
            # raw-ish fist helper (tip distance / palm length), helpful for tuning
            def _tip_norm(h: Optional[HandPosition]) -> str:
                if h is None or len(h.landmarks) < 21:
                    return "-"
                wrist = (h.landmarks[0].x_px, h.landmarks[0].y_px)
                palm = (h.landmarks[9].x_px, h.landmarks[9].y_px)
                palm_len = max(1.0, _dist(wrist, palm))
                cx, cy = h.center_px
                tip_idxs = [4, 8, 12, 16, 20]
                dsum = 0.0
                for i in tip_idxs:
                    dsum += _dist((h.landmarks[i].x_px, h.landmarks[i].y_px), (cx, cy))
                return f"{(dsum / len(tip_idxs)) / palm_len:0.2f}"
            l_tip = _tip_norm(l_hand)
            r_tip = _tip_norm(r_hand)

            def _score_norm(h: Optional[HandPosition]) -> str:
                if h is None or len(h.landmarks) < 21:
                    return "-"
                o = _hand_openness_score(h)
                wrist = (h.landmarks[0].x_px, h.landmarks[0].y_px)
                palm = (h.landmarks[9].x_px, h.landmarks[9].y_px)
                palm_len = max(1.0, _dist(wrist, palm))
                cx, cy = h.center_px
                tip_idxs = [4, 8, 12, 16, 20]
                dsum = 0.0
                for i in tip_idxs:
                    dsum += _dist((h.landmarks[i].x_px, h.landmarks[i].y_px), (cx, cy))
                tip_norm = (dsum / len(tip_idxs)) / palm_len
                return f"{_fist_score(openness=o, tip_norm=tip_norm):0.2f}"

            l_score = _score_norm(l_hand)
            r_score = _score_norm(r_hand)

            # Controls (right hand):
            # - point LEFT/RIGHT: choose active control (BPM vs KEY MODE)
            # - point UP/DOWN: adjust the active control
            # - close hand (fist): continue / start
            # Always render all detected hands with retro highlight + particles + L/R labels.
            for hnd in hands:
                _draw_hand_bbox_alpha(display, hnd, alpha=0.14)
                _draw_retro_hand(display, hnd)
                _draw_hand_label(display, hnd)
                _spawn_hand_particles(particles, hnd, n=14)

            # Soft cap particle count (prevents runaway buildup).
            if len(particles) > 1600:
                particles[:] = particles[-1200:]

            if rh is not None:
                direction = _point_direction(rh)
                trig = dir_gate.update_and_trigger(direction, stable_frames=3, cooldown_s=0.35)

                if state == "title":
                    if trig == "left":
                        active_panel = "bpm"
                    elif trig == "right":
                        active_panel = "mode"
                    elif trig == "up":
                        if active_panel == "bpm":
                            bpm_idx = int(np.clip(bpm_idx - 1, 0, len(bpm_values) - 1))
                        else:
                            mode_idx = 0  # Major
                    elif trig == "down":
                        if active_panel == "bpm":
                            bpm_idx = int(np.clip(bpm_idx + 1, 0, len(bpm_values) - 1))
                        else:
                            mode_idx = 1  # Minor

            else:
                dir_gate.last_dir = None
                dir_gate.stable_count = 0
                dir_gate.armed = True
                proceed_hold.reset()

            _update_and_draw_particles(display, particles, dt=dt)

            # Render UI by state
            if state == "welcome":
                _draw_rect_alpha(display, (0, 0, W, H), (0, 0, 0), alpha=0.35)

                t = time.time() - welcome_start_t
                logo_alpha = float(np.clip(t / 2.2, 0.0, 1.0))
                text_alpha = float(np.clip((t - 1.2) / 1.6, 0.0, 1.0))

                logo_h = min(240, max(140, int(H * 0.26)))
                logo = np.zeros((logo_h, W, 3), dtype=np.uint8)
                draw_ascii_banner(logo)
                _blend_image_alpha(display, logo, 0, int(H * 0.10), alpha=logo_alpha)

                prompt = "MAKE TWO FISTS TO PROCEED"
                (tw, _), _b = cv2.getTextSize(prompt, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
                px = max(20, (W - tw) // 2)
                py = int(H * 0.58)
                col = int(255 * text_alpha)
                cv2.putText(display, prompt, (px + 2, py + 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 6, cv2.LINE_AA)
                cv2.putText(display, prompt, (px, py), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (col, col, col), 3, cv2.LINE_AA)

                _draw_info_panel(
                    display,
                    info_rect,
                    bpm=bpm,
                    mode=mode,
                    extra_lines=[
                        ("State", "welcome"),
                        ("Hands", str(len(hands))),
                        ("2Fists", "yes" if two_fists else "no"),
                        ("L_fist", "yes" if l_fist else "no"),
                        ("R_fist", "yes" if r_fist else "no"),
                        ("L_open", l_open),
                        ("R_open", r_open),
                        ("L_tip", l_tip),
                        ("R_tip", r_tip),
                        ("L_C", l_score),
                        ("R_C", r_score),
                        ("Thres", f"{FIST_SCORE_THRESHOLD:0.2f}"),
                    ],
                )

                if did_proceed:
                    state = "title"
                    proceed_hold.reset()

            elif state == "title":
                bpm_items = [f"{b} BPM" for b in bpm_values]
                _draw_panel(display, bpm_rect, "BPM", active_panel == "bpm", bpm_idx, bpm_items)
                _draw_panel(display, mode_rect, "KEY MODE", active_panel == "mode", mode_idx, mode_items)
                _draw_info_panel(
                    display,
                    info_rect,
                    bpm=bpm,
                    mode=mode,
                    extra_lines=[
                        ("State", "setup"),
                        ("Hands", str(len(hands))),
                        ("2Fists", "yes" if two_fists else "no"),
                        ("L_fist", "yes" if l_fist else "no"),
                        ("R_fist", "yes" if r_fist else "no"),
                        ("L_open", l_open),
                        ("R_open", r_open),
                        ("L_tip", l_tip),
                        ("R_tip", r_tip),
                        ("L_C", l_score),
                        ("R_C", r_score),
                        ("Thres", f"{FIST_SCORE_THRESHOLD:0.2f}"),
                    ],
                )

                if did_proceed:
                    state = "countdown"
                    # Ensure countdown starts at Beat 1 (audio/UI share this clock).
                    if METRONOME_ENABLED:
                        metro.reset_phase()
                    metro.set_playback_enabled(False)
                    locked_instrument_idx = None
                    proceed_hold.reset()
            else:
                # Left instruments panel (semi-transparent)
                instr_rect = (pad, top, pad + int(W * 0.22), bottom)
                active_instr = instrument_idx if state == "countdown" else (locked_instrument_idx if locked_instrument_idx is not None else instrument_idx)
                _draw_instrument_panel(display, instr_rect, instruments, selected_idx=int(active_instr))

                # Chord ruler (Keys + Bass): to the left of the right info panel
                # IMPORTANT: chord selection (degree/pitch) is driven ONLY by the RIGHT hand.
                chords = _degree_chords(mode)
                # Display tonic at the bottom by reversing the list for the ruler.
                chords_ruler = list(reversed(chords))
                chord_w = int(max(140, W * 0.14))
                # Shrink vertical reach so it's easier to hit top/bottom without extreme hand travel.
                mid_h = max(1, bottom - top)
                chord_y0 = top + int(mid_h * 0.12)
                chord_y1 = bottom - int(mid_h * 0.16)
                chord_rect = (info_rect[0] - pad - chord_w, chord_y0, info_rect[0] - pad, chord_y1)
                chord_slot = chord_slot_state  # keep last known
                if rh_strict is not None:
                    cy = rh_strict.center_px[1]
                    inner_y0 = chord_rect[1] + 44
                    inner_y1 = chord_rect[3] - 10
                    # Natural mapping: hand up => selection up (toward the top of the ruler).
                    t = float(np.clip((cy - inner_y0) / max(1.0, (inner_y1 - inner_y0)), 0.0, 0.999999))
                    chord_slot = min(6, int(t * 7))
                    chord_slot_state = chord_slot
                if active_instr in (0, 2):
                    _draw_chord_ruler(display, chord_rect, chords_ruler, chord_slot)

                # Left hand controls instrument selection (point UP/DOWN)
                if state == "countdown" and lh is not None:
                    ldir = _point_direction(lh)
                    ltrig = left_dir_gate.update_and_trigger(
                        ldir,
                        stable_frames=INSTR_SELECT_STABLE_FRAMES,
                        cooldown_s=INSTR_SELECT_COOLDOWN_S,
                    )
                    if ltrig == "up":
                        instrument_idx = int(np.clip(instrument_idx - 1, 0, len(instruments) - 1))
                    elif ltrig == "down":
                        instrument_idx = int(np.clip(instrument_idx + 1, 0, len(instruments) - 1))
                else:
                    left_dir_gate.last_dir = None
                    left_dir_gate.stable_count = 0
                    left_dir_gate.armed = True

                # Keys chord hold + vibrato (loop only):
                # - closing left fist starts chord and holds it
                # - opening releases chord
                # - raising left hand above baseline adds vibrato (depth increases with height)
                if active_instr == 0 and state == "loop" and lh is not None and KEYS_ENABLED and metro_running:
                    l_val = float(_hand_openness_score(lh))

                    # open->closed triggers start
                    fired = keys_hold_gate.update_and_fire(
                        l_val,
                        close_th=KEYS_HOLD_CLOSE_TH,
                        open_th=KEYS_HOLD_OPEN_TH,
                        cooldown_s=0.12,
                    )
                    if fired:
                        keys_holding = True
                        keys_baseline_y = int(lh.center_px[1])
                        _, _lb, midi_notes = chords_ruler[chord_slot]
                        freqs = [_midi_to_hz(m) for m in midi_notes]
                        metro.start_hold_chord(freqs)
                        current_layer.append((metro.current_sample(), "chord_on", list(freqs)))

                    # closed->open releases
                    if keys_holding and (l_val >= KEYS_HOLD_OPEN_TH):
                        keys_holding = False
                        keys_baseline_y = None
                        metro.set_live_vibrato_cents(0.0)
                        metro.stop_hold_chord()
                        current_layer.append((metro.current_sample(), "chord_off", None))

                    # vibrato while holding: raise hand (smaller y) above baseline
                    if keys_holding and keys_baseline_y is not None:
                        dy = float(keys_baseline_y - int(lh.center_px[1]))
                        dy = max(0.0, dy)
                        depth = float(np.clip(dy / float(KEYS_VIBRATO_HEIGHT_PX_FOR_MAX), 0.0, 1.0)) * float(KEYS_VIBRATO_MAX_CENTS)
                        metro.set_live_vibrato_cents(depth)
                else:
                    # leaving keys: release any held chord and reset vibrato
                    if keys_holding and metro_running:
                        keys_holding = False
                        keys_baseline_y = None
                        metro.set_live_vibrato_cents(0.0)
                        metro.stop_hold_chord()
                    keys_hold_gate.is_closed = False

                # Bass trigger: same gesture + same ruler, but plays root note as bass (loop only)
                if active_instr == 2 and state == "loop" and lh is not None and BASS_ENABLED:
                    left_closed = _hand_openness_score(lh) <= 0.06
                    if left_fist_gate.rising_edge(left_closed, cooldown_s=0.18):
                        _rn, _lb, midi_notes = chords_ruler[chord_slot]
                        root = int(midi_notes[0]) - 24  # drop 2 octaves
                        root = int(np.clip(root, 24, 72))
                        if metro_running:
                            hz = _midi_to_hz(root)
                            metro.play_bass(hz)
                            current_layer.append((metro.current_sample(), "bass", float(hz)))
                else:
                    # don't clobber the gate if keys is active; only reset if neither uses it
                    if active_instr not in (0, 2):
                        left_fist_gate.prev = False

                # Kit triggers (loop only):
                # - left hand close => kick
                # - right hand close => snare
                if active_instr == 1 and state == "loop" and KIT_ENABLED and metro_running:
                    if lh is not None:
                        l_val = float(_hand_openness_score(lh))
                        if kit_kick_gate.update_and_fire(
                            l_val,
                            close_th=KIT_TRIGGER_CLOSE_TH,
                            open_th=KIT_TRIGGER_OPEN_TH,
                            cooldown_s=KIT_TRIGGER_COOLDOWN_S,
                        ):
                            metro.play_kick()
                            current_layer.append((metro.current_sample(), "kick", None))
                    else:
                        kit_kick_gate.is_closed = False

                    if r_hand is not None:
                        r_val = float(_hand_openness_score(r_hand))
                        if kit_snare_gate.update_and_fire(
                            r_val,
                            close_th=KIT_TRIGGER_CLOSE_TH,
                            open_th=KIT_TRIGGER_OPEN_TH,
                            cooldown_s=KIT_TRIGGER_COOLDOWN_S,
                        ):
                            metro.play_snare()
                            current_layer.append((metro.current_sample(), "snare", None))
                    else:
                        kit_snare_gate.is_closed = False
                else:
                    kit_kick_gate.is_closed = False
                    kit_snare_gate.is_closed = False

                # Keep the middle empty; status goes into the PROJECT panel on the right.
                extra: List[Tuple[str, str]] = []
                if state == "countdown":
                    # 2 bars in 4/4 => 8 beats
                    total_beats = 8
                    beats_elapsed = metro.beats_elapsed() if metro_running else 0
                    beats_left = max(0, total_beats - beats_elapsed)
                    big_n = max(1, beats_left)

                    # Big centered countdown number (no middle panel)
                    big = str(big_n)
                    (bw, _bh), _ = cv2.getTextSize(big, cv2.FONT_HERSHEY_SIMPLEX, 6.0, 12)
                    cx = (W - bw) // 2
                    cy = int(H * 0.52)
                    cv2.putText(display, big, (cx + 6, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 6.0, (0, 0, 0), 18, cv2.LINE_AA)
                    cv2.putText(display, big, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 6.0, (255, 255, 255), 12, cv2.LINE_AA)
                    extra.extend(
                        [
                            ("State", "countdown"),
                            ("Count", f"{big_n}"),
                            ("Bar", "0"),
                            ("Beat", str(metro.beat_in_bar())),
                            ("NextInstr", instruments[instrument_idx]),
                            ("Hands", str(len(hands))),
                            ("2Fists", "yes" if two_fists else "no"),
                            ("L_fist", "yes" if l_fist else "no"),
                            ("R_fist", "yes" if r_fist else "no"),
                            ("L_open", l_open),
                            ("R_open", r_open),
                            ("L_tip", l_tip),
                            ("R_tip", r_tip),
                            ("L_C", l_score),
                            ("R_C", r_score),
                            ("Thres", f"{FIST_SCORE_THRESHOLD:0.2f}"),
                        ]
                    )

                    if beats_left <= 0:
                        # Reset before entering loop so loop always starts at Bar 1 / Beat 1.
                        if METRONOME_ENABLED:
                            metro.reset_phase()
                        state = "loop"
                        locked_instrument_idx = instrument_idx
                        current_layer = []
                        metro.set_loop_layers(loop_layers)
                        metro.set_playback_enabled(True)
                        proceed_hold.reset()
                else:
                    # loop state (bar/beat derived from metronome clock)
                    bar = metro.bar_in_loop(8) if metro_running else 1
                    beat = metro.beat_in_bar() if metro_running else 1
                    beats_elapsed = metro.beats_elapsed() if metro_running else 0
                    extra.extend(
                        [
                            ("State", "loop1"),
                            ("Bar", str(bar)),
                            ("Beat", str(beat)),
                            ("Instr", instruments[int(active_instr)]),
                            ("Layers", str(len(loop_layers) + (1 if current_layer else 0))),
                            ("Hands", str(len(hands))),
                            ("2Fists", "yes" if two_fists else "no"),
                            ("L_fist", "yes" if l_fist else "no"),
                            ("R_fist", "yes" if r_fist else "no"),
                            ("L_open", l_open),
                            ("R_open", r_open),
                            ("L_tip", l_tip),
                            ("R_tip", r_tip),
                            ("L_C", l_score),
                            ("R_C", r_score),
                            ("Thres", f"{FIST_SCORE_THRESHOLD:0.2f}"),
                        ]
                    )

                    # Auto-transition: after 8 bars (32 beats), go to 8-beat countdown, and commit this layer.
                    if beats_elapsed >= 32:
                        if current_layer:
                            loop_layers.append(current_layer)
                            metro.set_loop_layers(loop_layers)
                            current_layer = []
                        metro.set_playback_enabled(False)
                        if METRONOME_ENABLED:
                            metro.reset_phase()
                        state = "countdown"
                        proceed_hold.reset()

                _draw_info_panel(display, info_rect, bpm=bpm, mode=mode, extra_lines=extra)

            # Banner & HUD
            if banner.shape[1] != W:
                banner = np.zeros((args.banner_height, W, 3), dtype=np.uint8)
                draw_ascii_banner(banner)
            hud = banner.copy()
            cv2.putText(
                hud,
                "point LEFT/RIGHT: choose control | point UP/DOWN: adjust | TWO fists (C>=thres): continue | q/esc quit",
                (14, hud.shape[0] - 12),
                cv2.FONT_HERSHEY_PLAIN,
                1.2,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                hud,
                f"fps: {fps:0.1f}",
                (W - 140, hud.shape[0] - 12),
                cv2.FONT_HERSHEY_PLAIN,
                1.2,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            out = np.zeros((hud.shape[0] + H, W, 3), dtype=np.uint8)
            out[: hud.shape[0], :, :] = hud
            out[hud.shape[0] :, :, :] = display

            cv2.imshow(window_name, out)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            prev_state = state

    cap.release()
    cv2.destroyAllWindows()
    metro.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


