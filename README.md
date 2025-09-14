# GigaVAD

Utilities for chunked speech transcription and voice activity detection.

## Features
- **VADProcessor**: wrapper around [Silero VAD](https://github.com/snakers4/silero-vad) for
  detecting speech segments.
- **ChunkingProcessor**: splits long audio into manageable chunks based on
  energy and optional overlap.
- **inference_gigaam.py**: command-line tool for transcribing long recordings
  with [GigaAM](https://github.com/salute-developers/GigaAM) models.
- **rupunct_apply.py**: optional script that restores Russian punctuation on
  JSONL segment files.

## Installation
Create a Python 3.11/3.12 virtual environment and install dependencies:

```bash
pip install -r requirements.txt  # if available
# or install manually
pip install torch soundfile
```

## Quick start
Transcribe a WAV file and obtain a JSON transcript:

```bash
python inference_gigaam.py input.wav transcript.json \
    --model v2_rnnt --lang ru --chunk_sec 22 --overlap_sec 1.0
```

To run plain VAD and save detected segments:

```bash
python vad_example.py input.wav vad_output.txt
```

For punctuation restoration of per-segment JSONL:

```bash
python rupunct_apply.py --in segments.jsonl --out segments_punct.jsonl \
    --merge_json transcript_punct.json
```

## Development
All modules contain docstrings and are intended to be easy to extend. Run
`python -m py_compile *.py` to validate syntax.

