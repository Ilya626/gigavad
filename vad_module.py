import numpy as np
from typing import List, Tuple
import torch


class VADProcessor:
    def __init__(
        self,
        threshold: float = 0.65,
        min_speech_ms: int = 200,
        min_silence_ms: int = 250,
        speech_pad_ms: int = 35,
        max_speech_duration_s: float = 22.0,
        use_cuda: bool = False,
    ):
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self.speech_pad_ms = speech_pad_ms
        self.max_speech_duration_s = max_speech_duration_s
        self.use_cuda = use_cuda
        self.model = self._load_model()

    def _load_model(self):
        try:
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                trust_repo=True
            )
            # Получаем функции из кортежа utils
            (get_speech_timestamps, _, _, _, _) = utils
            self.get_speech_timestamps = get_speech_timestamps
        except Exception as e:
            raise RuntimeError(f"Failed to load Silero VAD: {e}")
        
        device = 'cuda' if (self.use_cuda and torch.cuda.is_available()) else 'cpu'
        return model.to(device)

    def process(self, audio: np.ndarray, sr: int) -> List[Tuple[float, float]]:
        segs_raw = self._silero_vad_segments(audio, sr)
        return [(float(s["start"]), float(s["end"])) for s in segs_raw]

    def _silero_vad_segments(self, audio: np.ndarray, sr: int) -> List[Tuple[float, float]]:
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        
        wav_tensor = torch.from_numpy(audio.astype(np.float32)).to(self.model.device)
        
        return self.get_speech_timestamps(
            wav_tensor,
            self.model,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_ms,
            min_silence_duration_ms=self.min_silence_ms,
            speech_pad_ms=self.speech_pad_ms,
            max_speech_duration_s=self.max_speech_duration_s,
            return_seconds=True,
        )
