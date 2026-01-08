# ============================
# FILE: speech_analysis/audio_processing.py
# UPDATES: segments_raw, improved speech rate, improved filler counting, noise summary
# ============================

from __future__ import annotations

import os
import string
import tempfile
import shutil
import logging
import atexit
from pathlib import Path
from typing import Dict, List

from moviepy import VideoFileClip
import whisper_timestamped
import nltk
import wave
import librosa
import noisereduce as nr
import scipy.io.wavfile as wavf
import numpy as np
from nltk.tokenize import TreebankWordTokenizer
from pystoi import stoi  # NOTE: kept, but intelligibility will be revisited later

# =========================================================
# Model cache (project-local, agent-safe)
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_CACHE = PROJECT_ROOT / "data" / "cache" / "model_cache"
MODEL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(MODEL_CACHE))

# =========================================================
# Logging
# =========================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import re

_WORD_RE = re.compile(r"[A-Za-z]")

def normalize_token(t: str) -> str:
    t = (t or "").lower().strip()
    t = t.replace("’", "'")
    t = t.strip(string.punctuation)
    return t

def is_countable_word(t: str) -> bool:
    t = normalize_token(t)
    if not t:
        return False
    if _WORD_RE.search(t) is None:
        return False
    if len(t) == 1 and t not in {"a", "i"}:
        return False
    return True

def ensure_ffmpeg_on_path() -> None:
    import shutil
    from pathlib import Path

    if shutil.which("ffmpeg"):
        return

    try:
        import imageio_ffmpeg
        import shutil as _shutil

        ffmpeg_exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
        ffmpeg_dir = ffmpeg_exe.parent

        # Whisper expects ffmpeg.exe on PATH (Windows).
        # On Linux/Mac, ffmpeg_exe is already an executable; no copy needed,
        # but this doesn't hurt if it exists.
        target = ffmpeg_dir / "ffmpeg.exe"
        if not target.exists() and ffmpeg_exe.exists():
            try:
                _shutil.copyfile(ffmpeg_exe, target)
            except Exception:
                pass

        os.environ["PATH"] = str(ffmpeg_dir) + os.pathsep + os.environ.get("PATH", "")

        if shutil.which("ffmpeg") is None:
            raise RuntimeError(f"ffmpeg still not found after adding {ffmpeg_dir}")

        logger.info("ffmpeg available at: %s", shutil.which("ffmpeg"))

    except Exception as e:
        raise RuntimeError(
            "ffmpeg not found. Install system ffmpeg or `uv pip install imageio-ffmpeg`."
        ) from e

# =========================================================
# NLTK (download once)
# =========================================================
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# =========================================================
# Temp directory cleanup (important for notebooks)
# =========================================================
_TEMP_DIRS: List[str] = []

def _cleanup_temp_dirs():
    for d in _TEMP_DIRS:
        shutil.rmtree(d, ignore_errors=True)

atexit.register(_cleanup_temp_dirs)

# =========================================================
# Whisper model cache (in-process)
# =========================================================
_WHISPER_MODELS: Dict[str, object] = {}

def get_whisper_model(model_size: str):
    if model_size not in _WHISPER_MODELS:
        logger.info("Loading Whisper model: %s", model_size)
        _WHISPER_MODELS[model_size] = whisper_timestamped.load_model(model_size)
    return _WHISPER_MODELS[model_size]

# =========================================================
# Automatic Speech Recognition
# =========================================================
class AutomaticSpeechRecognition:
    def __init__(self, video_path: str, lang: str, whisper_model_size: str):
        self.temp_dir = tempfile.mkdtemp(prefix="audio_proc_")
        _TEMP_DIRS.append(self.temp_dir)

        audio_path = Path(self.temp_dir) / (Path(video_path).stem + ".wav")

        try:
            clip = VideoFileClip(video_path)
            clip.audio.write_audiofile(
                audio_path,
                codec="pcm_s16le",
                fps=16000,
                nbytes=2,
                ffmpeg_params=["-ac", "1"],
                logger=None,
            )
            self.duration = clip.duration
        except Exception:
            logger.exception("Audio extraction failed")
            raise

        self.audio_path = str(audio_path)
        self.lang = lang
        self.model_size = whisper_model_size
        self.transcription = None

    def transcribe(self):
        ensure_ffmpeg_on_path()
        model = get_whisper_model(self.model_size)
        audio = whisper_timestamped.load_audio(self.audio_path)

        self.transcription = whisper_timestamped.transcribe(
            model,
            audio,
            language=self.lang,
            detect_disfluencies=True,
            remove_punctuation_from_words=False,
        )

        self._trim_invalid_segments()

        # Keep BOTH cleaned full text + raw segments
        cleaned_text = self._clean_text(self.transcription.get("text", ""))

        words, noise = self._split_words_and_noise()

        segments_raw = list(self.transcription.get("segments", []))
        segments_compact = [
            {"start": seg.get("start", 0.0), "end": seg.get("end", 0.0), "text": seg.get("text", "")}
            for seg in segments_raw
        ]

        return cleaned_text, words, noise, segments_compact, segments_raw

    def _clean_text(self, text: str) -> str:
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation + string.digits))
        return " ".join(text.split())

    def _split_words_and_noise(self):
        words, noise = [], []
        for seg in self.transcription.get("segments", []):
            for w in seg.get("words", []):
                if w.get("text") == "[*]":
                    noise.append((float(w.get("start", 0.0)), float(w.get("end", 0.0))))
                else:
                    words.append(w)
        # Keep words sorted by start (defensive)
        try:
            words.sort(key=lambda x: float(x.get("start", 0.0)))
        except Exception:
            pass
        return words, noise

    def _trim_invalid_segments(self):
        segments = self.transcription.get("segments", [])
        valid = []
        for seg in segments:
            if seg.get("end", 0.0) <= self.duration:
                valid.append(seg)
            else:
                break
        self.transcription["segments"] = valid

# =========================================================
# Analysis components
# =========================================================
class FillerWordsAndPhrases:
    EN = {
        "words": {
            "um", "uh", "erm", "ah", "hmm",
            "like", "so", "just", "actually", "basically", "really",
            "yeah", "ok", "okay", "right", "well",
            "yknow", "youknow",
        },
        "phrases": {
            "you know", "i mean", "kind of", "sort of", "you see", "let's say",
        },
    }

    def __init__(self, words_with_timestamps: List[Dict]):
        # words_with_timestamps: list of {"text": "...", "start": 0.0, "end": 0.0}
        self.words = words_with_timestamps

    def count(self):
        """
        Returns:
            word_occurrences: List of {word, start, end}
            phrase_occurrences: List of {phrase, start, end}
            stats: {word_counts: {}, phrase_counts: {}}
        """
        word_occurrences = []
        phrase_occurrences = []
        
        word_counts: Dict[str, int] = {}
        phrase_counts: Dict[str, int] = {}

        # 1. Analyze single words
        for w in self.words:
            raw_text = w.get("text", "")
            t_norm = str(raw_text).lower().replace("’", "'").strip()
            # remove punctuation for checking
            t_check = t_norm.strip(string.punctuation)
            
            if t_check in self.EN["words"]:
                word_counts[t_check] = word_counts.get(t_check, 0) + 1
                word_occurrences.append({
                    "text": t_check,
                    "start": w.get("start"),
                    "end": w.get("end")
                })

        # 2. Analyze phrases (bigrams)
        # We need to look at adjacent words. 
        # Note: self.words should be sorted by time.
        n = len(self.words)
        for i in range(n - 1):
            w1 = self.words[i]
            w2 = self.words[i+1]
            
            t1 = str(w1.get("text", "")).lower().strip(string.punctuation)
            t2 = str(w2.get("text", "")).lower().strip(string.punctuation)
            
            phrase = f"{t1} {t2}"
            if phrase in self.EN["phrases"]:
                phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
                phrase_occurrences.append({
                    "text": phrase,
                    "start": w1.get("start"),
                    "end": w2.get("end")
                })

        return {
            "occurrences": {
                "words": word_occurrences,
                "phrases": phrase_occurrences,
            },
            "counts": {
                "words": word_counts,
                "phrases": phrase_counts
            }
        }

class SpeechRate:
    def __init__(self, words, window_sec=30, slow_wpm=110, fast_wpm=170, min_window_sec=10):
        self.words = words
        self.window_sec = window_sec
        self.slow_wpm = slow_wpm
        self.fast_wpm = fast_wpm
        self.min_window_sec = min_window_sec

    def analyze(self):
        """
        Returns:
          - windows: [{start,end,wpm,word_count}, ...]
          - slow_intervals: [[start,end], ...]
          - fast_intervals: [[start,end], ...]
        """
        if not self.words:
            return [], [], []

        try:
            t0 = float(self.words[0]["start"])
            t_end = float(self.words[-1]["end"])
        except Exception:
            return [], [], []

        windows = []
        slow, fast = [], []

        cur_start = t0
        # We assume words are sorted by start
        wi = 0
        n = len(self.words)

        while cur_start < t_end:
            cur_end = min(cur_start + self.window_sec, t_end)
            dur = cur_end - cur_start
            if dur < self.min_window_sec:
                break

            # count words overlapping [cur_start,cur_end)
            # advance pointer to first word that might overlap
            while wi < n and float(self.words[wi].get("end", 0.0)) <= cur_start:
                wi += 1

            count = 0
            wj = wi
            while wj < n and float(self.words[wj].get("start", 0.0)) < cur_end:
                tok = str(self.words[wj].get("text", ""))
                if is_countable_word(tok):
                    count += 1
                wj += 1


            wpm = (count / dur) * 60.0 if dur > 0 else 0.0
            windows.append({
                "start": round(cur_start, 3),
                "end": round(cur_end, 3),
                "wpm": round(wpm, 1),
                "word_count": int(count),
            })

            if wpm < self.slow_wpm:
                slow.append([round(cur_start, 3), round(cur_end, 3)])
            elif wpm > self.fast_wpm:
                fast.append([round(cur_start, 3), round(cur_end, 3)])

            cur_start = cur_end

        return windows, slow, fast

class BackgroundNoise:
    def __init__(self, noise, window=30, threshold=0.45):
        self.noise = noise
        self.window = window
        self.threshold = threshold

    def analyze(self):
        # (unchanged: keep your interval detector)
        results = []
        if not self.noise:
            return results

        start, total = 0, 0.0
        for end in range(len(self.noise)):
            total += self.noise[end][1] - self.noise[end][0]
            while self.noise[end][1] - self.noise[start][0] > self.window:
                total -= self.noise[start][1] - self.noise[start][0]
                start += 1
            period = self.noise[end][1] - self.noise[start][0]
            if period and total / period > self.threshold:
                results.append([self.noise[start][0], self.noise[end][1]])
        return results

class IntelligibilityFromASR:
    def __init__(
        self,
        segments_raw,
        low_conf_threshold: float = 0.55,
        nospeech_threshold: float = 0.6,
    ):
        self.segments_raw = segments_raw or []
        self.low_conf_threshold = low_conf_threshold
        self.nospeech_threshold = nospeech_threshold

    @staticmethod
    def _clamp01(x: float) -> float:
        return max(0.0, min(1.0, x))

    @staticmethod
    def _normalize_avg_logprob(avg_logprob: float) -> float:
        """
        avg_logprob is typically negative. Roughly:
          -1.0 (good) to -3.0 (bad) depending on audio/model.
        Map [-3.0, -1.0] -> [0,1].
        """
        if avg_logprob is None:
            return 0.5
        try:
            x = float(avg_logprob)
        except Exception:
            return 0.5
        # map -3 -> 0, -1 -> 1
        return IntelligibilityFromASR._clamp01((x - (-3.0)) / ((-1.0) - (-3.0)))

    def compute(self):
        per_segment = []
        low_conf_intervals = []

        for seg in self.segments_raw:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            conf = seg.get("confidence", None)
            nsp = seg.get("no_speech_prob", None)
            alp = seg.get("avg_logprob", None)

            # confidence sometimes missing or not 0..1; guard it
            try:
                conf_f = float(conf) if conf is not None else None
            except Exception:
                conf_f = None

            try:
                nsp_f = float(nsp) if nsp is not None else 0.0
            except Exception:
                nsp_f = 0.0

            alp_score = self._normalize_avg_logprob(alp)

            # Combine signals:
            # - main weight: confidence
            # - secondary: avg_logprob normalization
            # - penalty: no_speech_prob (if high, likely silence or non-speech)
            if conf_f is None:
                score = 0.6 * alp_score + 0.4 * (1.0 - self._clamp01(nsp_f))
            else:
                score = 0.7 * self._clamp01(conf_f) + 0.3 * alp_score
                # penalty for likely no-speech
                if nsp_f >= self.nospeech_threshold:
                    score *= 0.6

            score = self._clamp01(score)

            per_segment.append({
                "start": round(start, 3),
                "end": round(end, 3),
                "score": round(score, 3),
                "confidence": None if conf_f is None else round(self._clamp01(conf_f), 3),
                "avg_logprob": None if alp is None else float(alp),
                "no_speech_prob": round(self._clamp01(nsp_f), 3),
            })

            if score < self.low_conf_threshold:
                low_conf_intervals.append([round(start, 3), round(end, 3)])

        # global score: duration-weighted average
        total_dur = 0.0
        weighted = 0.0
        for s in per_segment:
            dur = max(0.0, s["end"] - s["start"])
            total_dur += dur
            weighted += dur * s["score"]
        global_score = (weighted / total_dur) if total_dur > 0 else 0.0

        return {
            "global_score": round(float(global_score), 3),
            "per_segment": per_segment,
            "low_confidence_intervals": low_conf_intervals,
        }

# =========================================================
# 🎯 Final SpeechProcessingSubsystem
# =========================================================
class SpeechProcessingSubsystem:
    def __init__(
        self,
        video_path: str,
        lang: str = "en",
        whisper_model_size: str = "small",
        intelligibility_segment_len: int = 10,
    ):
        asr = AutomaticSpeechRecognition(video_path, lang, whisper_model_size)
        text, words, noise, segments_compact, segments_raw = asr.transcribe()

        self.segments = segments_compact
        self.segments_raw = segments_raw
        self.text = text
        self.words = words
        self.noise = noise
        self.wav_path = asr.audio_path
        self.seg_len = intelligibility_segment_len

    def run(self) -> Dict:
        # Pass full word objects with timestamps
        fillers = FillerWordsAndPhrases(words_with_timestamps=self.words)
        filler_stats = fillers.count()

        rate_windows, slow, fast = SpeechRate(self.words).analyze()

        noise_intervals = BackgroundNoise(self.noise).analyze()
        total_noise = float(sum((b - a) for a, b in self.noise)) if self.noise else 0.0
        duration = float(self.words[-1]["end"] - self.words[0]["start"]) if self.words else 0.0
        noise_fraction = (total_noise / duration) if duration > 0 else 0.0

        intel = IntelligibilityFromASR(self.segments_raw).compute()

        return {
            "transcription": self.text,
            "segments": self.segments,               # compact for alignment
            "segments_raw": self.segments_raw,       # full whisper segments for confidence later
            "filler_words": filler_stats["counts"]["words"],
            "filler_phrases": filler_stats["counts"]["phrases"],
            "filler_occurrences": filler_stats["occurrences"], # Added for per-slide feedback
            "speech_rate": {
                "windows": rate_windows,
                "slow": slow,
                "fast": fast,
            },
            "background_noise": {
                "intervals": noise_intervals,
                "total_noise_sec": round(total_noise, 2),
                "fraction": round(noise_fraction, 3),
            },
            "intelligibility": intel,
        }
