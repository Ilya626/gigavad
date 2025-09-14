import sys
from pathlib import Path
import soundfile as sf
from vad_module import VADProcessor

def main(input_wav: str, output_txt: str):
    try:
        # Проверка существования входного файла
        input_path = Path(input_wav)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file {input_wav} not found")
        
        # Load audio
        audio, sr = sf.read(input_wav)
        print(f"Loaded audio: {len(audio)} samples at {sr}Hz")
        
        # Process with VAD
        vad = VADProcessor(
            threshold=0.3,
            min_speech_ms=100,
            min_silence_ms=200,
            speech_pad_ms=30,
            use_cuda=False
        )
        segments = vad.process(audio, sr)
        print(f"Detected {len(segments)} speech segments")
        
        # Write results
        output_path = Path(output_txt)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with output_path.open('w', encoding='utf-8') as f:
            for start, end in segments:
                f.write(f"{start:.1f}-{end:.1f} Голос\n")
            f.write("ОСТАЛЬНОЕ ТИШИНА\n")
        
        print(f"Results successfully written to {output_path}")
        return True
    except Exception as e:
        print(f"Error processing VAD: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python vad_example.py input.wav output.txt")
        sys.exit(1)
        
    main(sys.argv[1], sys.argv[2])
    print(f"VAD results saved to {sys.argv[2]}")
    print("Compare with: diff -y vad_output.txt vad_hand_results_6m.txt")
