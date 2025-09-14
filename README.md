# GigaVAD

Utilities for chunked speech transcription and voice activity detection.

## Features
- **VADProcessor**: wrapper around [Silero VAD](https://github.com/snakers4/silero-vad) for
  detecting speech segments. Supports an optional ``model_dir`` parameter to
  load pre-downloaded weights and avoid using ``torch.hub``.
- **ChunkingProcessor**: splits long audio into manageable chunks based on
  energy and optional overlap. Supports adaptive silence thresholds based on
  noise-floor statistics which can be disabled with ``adaptive=False`` or the
  ``--no_adaptive`` CLI flag.
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
# add --vad_silero --silero_model_dir /path/to/silero-vad to use local VAD
# use --vad_pad_ms to extend VAD segments and avoid cutting words at boundaries
```

Increasing `--vad_pad_ms` adds the specified milliseconds of context before and
after every VAD segment prior to bin packing. This extra margin helps prevent
words from being truncated at chunk boundaries, improving boundary safety.

Use `--vad_min_bin_speech` to require a minimum amount of speech in each VAD
bin. If a gap or duration limit would close the bin early, segments are merged
with subsequent ones until this threshold is reached.

Pass `--vad_merge_segs` to merge adjacent VAD segments separated by short
pauses. Gaps shorter than `--vad_min_gap_sec` are joined before bin packing,
reducing excessive fragmentation of speech.

To run plain VAD and save detected segments:

```bash
# optionally provide a directory with silero-vad model.jit and utils_vad.py
python vad_example.py input.wav vad_output.txt [model_dir]
```

For punctuation restoration of per-segment JSONL:

```bash
python rupunct_apply.py --in segments.jsonl --out segments_punct.jsonl \
    --merge_json transcript_punct.json
```

## Development
All modules contain docstrings and are intended to be easy to extend. Run
`python -m py_compile *.py` to validate syntax.

