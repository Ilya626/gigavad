import numpy as np
from typing import List, Tuple, Optional
import torch

class VADProcessor:
    def __init__(self, threshold=0.5, min_speech_ms=150, min_silence_ms=300, 
                 speech_pad_ms=50, use_cuda=False):
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self.speech_pad_ms = speech_pad_ms
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
        return self._silero_vad_segments(audio, sr)

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
            return_seconds=True
        )
