# speaker_trainer/speech_analysis/audio_processing.py

from __future__ import annotations

# ---------- stdlib ----------
import os
import string
import tempfile
import shutil
import logging
import atexit
from pathlib import Path
from typing import Dict, List, Tuple

from concurrent.futures import ProcessPoolExecutor

# ---------- third-party ----------
from moviepy import VideoFileClip
import whisper_timestamped
import nltk
import wave
import librosa
import noisereduce as nr
import scipy.io.wavfile as wavf
import numpy as np
from pystoi import stoi
from nltk.tokenize import TreebankWordTokenizer


# =========================================================
# Model cache (project-local, agent-safe)
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_CACHE = PROJECT_ROOT / "model_cache"
MODEL_CACHE.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("XDG_CACHE_HOME", str(MODEL_CACHE))


# =========================================================
# Logging
# =========================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_ffmpeg_on_path() -> None:
    import os
    import shutil
    from pathlib import Path

    if shutil.which("ffmpeg"):
        return

    try:
        import imageio_ffmpeg
        import shutil as _shutil

        ffmpeg_exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
        ffmpeg_dir = ffmpeg_exe.parent

        # Whisper expects ffmpeg.exe on PATH
        target = ffmpeg_dir / "ffmpeg.exe"
        if not target.exists():
            _shutil.copyfile(ffmpeg_exe, target)

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
        except Exception as e:
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

        cleaned_text = self._clean_text(self.transcription["text"])
        words, noise = self._split_words_and_noise()

        segments = [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
            }
            for seg in self.transcription.get("segments", [])
        ]

        return cleaned_text, words, noise, segments

    def _clean_text(self, text: str) -> str:
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation + string.digits))
        return " ".join(text.split())

    def _split_words_and_noise(self):
        words, noise = [], []
        for seg in self.transcription.get("segments", []):
            for w in seg.get("words", []):
                if w["text"] == "[*]":
                    noise.append((w["start"], w["end"]))
                else:
                    words.append(w)
        return words, noise

    def _trim_invalid_segments(self):
        segments = self.transcription["segments"]
        valid = []
        for seg in segments:
            if seg["end"] <= self.duration:
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
            "um","uh","like","so","just","actually",
            "basically","really","yeah","okay","right","y'know"
        },
        "phrases": {
            "you know","i mean","you see",
            "let's say","kind of","sort of"
        },
    }

    def __init__(self, text: str):
        self.tokens = TreebankWordTokenizer().tokenize(text.lower())

    def count(self):
        word_counts = {}
        phrase_counts = {}

        for t in self.tokens:
            if t in self.EN["words"]:
                word_counts[t] = word_counts.get(t, 0) + 1

        for i in range(len(self.tokens) - 1):
            phrase = f"{self.tokens[i]} {self.tokens[i+1]}"
            if phrase in self.EN["phrases"]:
                phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

        return word_counts, phrase_counts


class SpeechRate:
    WINDOW = 60
    MIN = 60
    MAX = 140

    def __init__(self, words):
        self.words = words

    def analyze(self):
        slow, fast = [], []
        if not self.words:
            return slow, fast

        start = self.words[0]["start"]
        count = 0

        for w in self.words:
            count += 1
            if w["end"] - start >= self.WINDOW:
                if count < self.MIN:
                    slow.append([start, w["end"]])
                elif count > self.MAX:
                    fast.append([start, w["end"]])
                start = w["start"]
                count = 1

        return slow, fast


class BackgroundNoise:
    def __init__(self, noise, window=30, threshold=0.45):
        self.noise = noise
        self.window = window
        self.threshold = threshold

    def analyze(self):
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


class Intelligibility:
    def __init__(self, wav_path: str, segment_len: int = 10):
        self.wav_path = wav_path
        self.segment_len = segment_len

    @staticmethod
    def _stoi_segment(args):
        import os
        import wave
        import librosa
        import noisereduce as nr
        import scipy.io.wavfile as wavf
        import numpy as np
        from pystoi import stoi

        wav_path, i, seg_len = args

        clip = wave.open(wav_path)
        duration = clip.getnframes() / clip.getframerate()

        start = i * seg_len
        end = min((i + 1) * seg_len, duration)

        data, rate = librosa.load(wav_path, offset=start, duration=end - start)
        reduced = nr.reduce_noise(y=data, sr=rate)

        tmp = f"tmp_{os.getpid()}_{i}.wav"
        wavf.write(tmp, rate, reduced)

        clean, _ = librosa.load(tmp)
        orig, _ = librosa.load(wav_path, offset=start, duration=end - start)

        try:
            os.remove(tmp)
        except Exception:
            pass

        return i, float(np.round(stoi(clean, orig, rate), 3))

    def compute(self):
        import wave
        import numpy as np

        clip = wave.open(self.wav_path)
        duration = clip.getnframes() / clip.getframerate()
        segments = int(np.ceil(duration / self.segment_len))
        scores = np.zeros(segments)

        # Windows-safe: sequential by default
        args = [(self.wav_path, i, self.segment_len) for i in range(segments)]
        for i, s in map(Intelligibility._stoi_segment, args):
            scores[i] = s

        return scores



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
        text, words, noise, segments = asr.transcribe()

        self.segments = segments
        self.text = text
        self.words = words
        self.noise = noise
        self.wav_path = asr.audio_path
        self.seg_len = intelligibility_segment_len

    def run(self) -> Dict:
        fillers = FillerWordsAndPhrases(self.text)
        slow, fast = SpeechRate(self.words).analyze()
        noise_intervals = BackgroundNoise(self.noise).analyze()
        intelligibility = Intelligibility(self.wav_path, self.seg_len).compute()

        return {
            "transcription": self.text,
            "segments": self.segments,
            "filler_words": fillers.count()[0],
            "filler_phrases": fillers.count()[1],
            "speech_rate": {
                "slow": slow,
                "fast": fast,
            },
            "background_noise": noise_intervals,
            "intelligibility": intelligibility.tolist(),
        }

