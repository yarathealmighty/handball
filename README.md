# Handball Shot Timestamp Extractor

A Python CLI tool that detects handball shots in a game video and outputs a CSV of timestamps. It can use a Hugging Face action recognition model when available, falling back to a heuristic motion+audio method otherwise.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On macOS, if `ffmpeg` is missing, install it via Homebrew:

```bash
brew install ffmpeg
```

## Usage

```bash
python detect_handball_shots.py \
  --video /path/to/game.mp4 \
  --out shots.csv \
  --method auto \
  --hf-model facebook/timesformer-base-finetuned-k400  \
  --threshold 0.6
```

- `--method`: `auto` (default), `hf`, or `heuristic`.
- `--hf-model`: Hugging Face action model id. Default: `facebook/timesformer-base-finetuned-k400`.
- `--threshold`: Confidence threshold for model probabilities.
- `--frame-step`: Sample every N frames for speed (heuristic and HF pre-sampling).

The CSV contains:

```
timestamp_seconds,label,confidence,source
```

## Notes
- For best results, provide broadcast footage with a scoreboard shot clock visible.
- The heuristic method uses motion intensity and audio onset peaks; it is a coarse fallback.

## Limitations
- Off-the-shelf action models may not perfectly capture "handball shot" specifically; you can adjust `--threshold` and smoothing parameters.
- Consider fine-tuning on labeled handball data for higher accuracy.
