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
  with [GigaAM](https://github.com/salute-developers/GigaAM) models. CLI options
  are grouped (Chunking, VAD, etc.) and can be loaded from a JSON config file
  via the `--config` flag.
- **rupunct_apply.py**: optional script that restores Russian punctuation on
  JSONL segment files.

## Installation
Create a Python 3.11/3.12 virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

## Quick start
Transcribe a WAV file and obtain a JSON transcript:

```bash
# direct CLI usage
python inference_gigaam.py input.wav transcript.json \
    --model v2_rnnt --lang ru --chunk_sec 22 --overlap_sec 1.0

# or place arguments in a JSON config file
echo '{"model": "v2_rnnt", "lang": "ru", "chunk_sec": 22, "overlap_sec": 1.0}' > cfg.json
python inference_gigaam.py input.wav transcript.json --config cfg.json

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
python vad_example.py input.wav vad_output.txt
```

For punctuation restoration of per-segment JSONL:

```bash
python rupunct_apply.py --in segments.jsonl --out segments_punct.jsonl \
    --merge_json transcript_punct.json
```

## Offline Silero VAD

Download `model.jit` and `utils_vad.py` from the
[snakers4/silero-vad](https://github.com/snakers4/silero-vad) repository and
place them in a directory. Provide this directory to run VAD without relying on
`torch.hub`:

```bash
python vad_example.py input.wav vad_output.txt --silero_model_dir /path/to/silero-vad
python inference_gigaam.py input.wav transcript.json \
    --model v2_rnnt --lang ru --vad_silero \
    --silero_model_dir /path/to/silero-vad
```

## Testing

The repository includes basic unit tests and example commands. Example WAV and
reference text files (e.g. `ilya_6m_test.wav` and `vad_hand_results_6m.txt`)
are user-provided.

1. Run automated tests:

   ```bash
   python -m pytest tests
   ```

2. Execute VAD on a sample file and compare against a reference:

   ```bash
   python vad_example.py ilya_6m_test.wav vad_output.txt --silero_model_dir /path/to/silero-vad
   diff -y vad_output.txt vad_hand_results_6m.txt
   ```

3. Transcribe the sample file with bin packing and optional VAD:

   ```bash
   python inference_gigaam.py ilya_6m_test.wav transcript.json \
       --model v2_rnnt --lang ru --chunk_sec 22 --overlap_sec 1.0 \
       --vad_silero --silero_model_dir /path/to/silero-vad
   ```

   - `--vad_silero` toggles Silero-based VAD (omit to disable).
   - `--chunk_sec` and `--overlap_sec` control bin packing of speech segments.
   - PyTorch uses the GPU automatically when available.

## Development
All modules contain docstrings and are intended to be easy to extend. Run
`python -m py_compile *.py` to validate syntax.

