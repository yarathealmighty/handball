import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional heavy deps loaded lazily when needed

def try_import_transformers():
	try:
		from transformers import pipeline  # type: ignore
		return pipeline
	except Exception:
		return None


def try_import_torch():
	try:
		import torch  # type: ignore
		return torch
	except Exception:
		return None


def try_import_cv2():
	try:
		import cv2  # type: ignore
		return cv2
	except Exception:
		return None


def try_import_librosa():
	try:
		import librosa  # type: ignore
		return librosa
	except Exception:
		return None


def try_import_pil():
	try:
		from PIL import Image  # type: ignore
		return Image
	except Exception:
		return None


@dataclass
class Detection:
	timestamp_seconds: float
	label: str
	confidence: float
	source: str  # "hf" or "heuristic"


def read_video_metadata(video_path: str) -> Tuple[int, float, int, int]:
	cv2 = try_import_cv2()
	if cv2 is None:
		raise RuntimeError("opencv-python is required. Please install dependencies.")
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		raise RuntimeError(f"Failed to open video: {video_path}")
	fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
	n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
	cap.release()
	return n_frames, float(fps), width, height


def sample_video_frames(video_path: str, frame_step: int = 10, max_frames: Optional[int] = None) -> List[Tuple[np.ndarray, float, int]]:
	cv2 = try_import_cv2()
	if cv2 is None:
		raise RuntimeError("opencv-python is required. Please install dependencies.")
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		raise RuntimeError(f"Failed to open video: {video_path}")
	fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
	frames: List[Tuple[np.ndarray, float, int]] = []
	frame_index = 0
	kept = 0
	while True:
		ret, frame = cap.read()
		if not ret:
			break
		if frame_index % frame_step == 0:
			# timestamp in seconds
			sec = frame_index / float(fps)
			frames.append((frame, sec, frame_index))
			kept += 1
			if max_frames is not None and kept >= max_frames:
				break
		frame_index += 1
	cap.release()
	return frames


def extract_clip_around_index(video_path: str, center_index: int, num_frames: int = 8, stride: int = 2) -> List["Image.Image"]:
	cv2 = try_import_cv2()
	Image = try_import_pil()
	if cv2 is None:
		raise RuntimeError("opencv-python is required.")
	if Image is None:
		raise RuntimeError("Pillow is required. Please install pillow.")
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		raise RuntimeError(f"Failed to open video: {video_path}")
	start = max(0, center_index - (num_frames // 2) * stride)
	frames: List["Image.Image"] = []
	for i in range(num_frames):
		idx = start + i * stride
		cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
		ret, frame = cap.read()
		if not ret:
			break
		# BGR -> RGB -> PIL
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frames.append(Image.fromarray(rgb))
	cap.release()
	return frames


def run_hf_action_detection(video_path: str, model_id: str, threshold: float = 0.6, frame_step: int = 10) -> List[Detection]:
	pipeline = try_import_transformers()
	if pipeline is None:
		raise RuntimeError("transformers is not installed or failed to import.")
	# Strictly use video-classification pipeline and pass a dict with "video": clip
	try:
		clf = pipeline("video-classification", model=model_id, top_k=None)
	except Exception as e:
		raise RuntimeError(f"Failed to init video-classification pipeline for {model_id}: {e}")

	detections: List[Detection] = []
	frames = sample_video_frames(video_path, frame_step=frame_step)
	for _, ts, frame_idx in frames:
		clip = extract_clip_around_index(video_path, frame_idx, num_frames=8, stride=2)
		if not clip:
			continue
		result = clf({"video": clip})
		# result: list of dicts [{'label': 'class', 'score': 0.9}, ...]
		if isinstance(result, list):
			for item in result:
				label = str(item.get("label", ""))
				score = float(item.get("score", 0.0))
				if score >= threshold and is_handball_shot_label(label):
					detections.append(Detection(ts, label, score, "hf"))
	return detections


def is_handball_shot_label(label: str) -> bool:
	label_l = label.lower()
	keywords = [
		"handball",
		"throw",
		"shooting",
		"shot",
		"goal shot",
		"overarm throw",
	]
	return any(k in label_l for k in keywords)


def heuristic_motion_audio(video_path: str, frame_step: int = 2, motion_thresh: float = 0.15, audio_prominence: float = 0.8) -> List[Detection]:
	cv2 = try_import_cv2()
	librosa = try_import_librosa()
	if cv2 is None:
		raise RuntimeError("opencv-python is required for heuristic method.")

	# Motion intensity via frame differencing
	cap = cv2.VideoCapture(video_path)
	fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
	prev_gray = None
	motion_times: List[float] = []
	idx = 0
	while True:
		ret, frame = cap.read()
		if not ret:
			break
		if idx % frame_step == 0:
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			gray = cv2.GaussianBlur(gray, (5, 5), 0)
			if prev_gray is not None:
				diff = cv2.absdiff(gray, prev_gray)
				intensity = float(np.mean(diff) / 255.0)
				if intensity > motion_thresh:
					motion_times.append(idx / float(fps))
			prev_gray = gray
		idx += 1
	cap.release()

	# Audio onsets / peaks around crowd noise or commentator shouts
	audio_peaks: List[float] = []
	if librosa is not None:
		try:
			y, sr = librosa.load(video_path, sr=None)  # requires ffmpeg
			onset_env = librosa.onset.onset_strength(y=y, sr=sr)
			onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, backtrack=False)
			# Normalize prominence by percentile
			if len(onset_env) > 0:
				thr = np.quantile(onset_env, audio_prominence)
				for f in onset_frames:
					if onset_env[min(f, len(onset_env)-1)] >= thr:
						audio_peaks.append(float(librosa.frames_to_time([f], sr=sr)[0]))
		except Exception:
			# Audio extraction may fail; ignore audio
			pass

	# Combine events: near-coincident motion and audio within a window
	detections: List[Detection] = []
	if audio_peaks:
		for t in motion_times:
			if any(abs(t - a) <= 0.6 for a in audio_peaks):
				detections.append(Detection(t, "handball_shot_candidate", 0.5, "heuristic"))
	else:
		# If no audio, keep top-K motion spikes by spacing
		motion_times_sorted = sorted(motion_times)
		last = -10.0
		for t in motion_times_sorted:
			if t - last >= 1.0:
				detections.append(Detection(t, "handball_shot_candidate", 0.4, "heuristic"))
				last = t
	return detections


def deduplicate_and_sort(dets: List[Detection], merge_window: float = 1.0) -> List[Detection]:
	# Merge close detections, keeping the one with higher confidence and preferring HF
	dets_sorted = sorted(dets, key=lambda d: d.timestamp_seconds)
	merged: List[Detection] = []
	for d in dets_sorted:
		if not merged:
			merged.append(d)
			continue
		if d.timestamp_seconds - merged[-1].timestamp_seconds <= merge_window:
			# choose better
			best = merged[-1]
			if (d.source == "hf" and best.source != "hf") or (d.confidence > best.confidence):
				merged[-1] = d
		else:
			merged.append(d)
	return merged


def write_csv(dets: List[Detection], out_path: str) -> None:
	os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
	with open(out_path, "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["timestamp_seconds", "label", "confidence", "source"])
		for d in dets:
			writer.writerow([f"{d.timestamp_seconds:.3f}", d.label, f"{d.confidence:.3f}", d.source])


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Detect handball shots and export CSV timestamps")
	p.add_argument("--video", required=True, help="Path to input video")
	p.add_argument("--out", required=True, help="Path to output CSV")
	p.add_argument("--method", default="auto", choices=["auto", "hf", "heuristic"], help="Detection method")
	p.add_argument("--hf-model", default="facebook/timesformer-base-finetuned-k400", help="Hugging Face model id")
	p.add_argument("--threshold", type=float, default=0.6, help="Confidence threshold for model")
	p.add_argument("--frame-step", type=int, default=10, help="Frame sampling step for speed")
	return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
	args = parse_args(argv)
	if not os.path.exists(args.video):
		print(f"Video not found: {args.video}", file=sys.stderr)
		return 2

	all_dets: List[Detection] = []
	if args.method in ("auto", "hf"):
		try:
			hf_dets = run_hf_action_detection(args.video, args.hf_model, args.threshold, args.frame_step)
			all_dets.extend(hf_dets)
		except Exception as e:
			if args.method == "hf":
				print(f"HF detection failed: {e}", file=sys.stderr)
				return 1
			print(f"HF detection unavailable, falling back to heuristic: {e}")

	if args.method in ("auto", "heuristic") and not all_dets:
		try:
			heu_dets = heuristic_motion_audio(args.video, frame_step=max(2, args.frame_step // 2))
			all_dets.extend(heu_dets)
		except Exception as e:
			print(f"Heuristic detection failed: {e}", file=sys.stderr)
			if args.method == "heuristic":
				return 1

	final_dets = deduplicate_and_sort(all_dets)
	write_csv(final_dets, args.out)
	print(f"Wrote {len(final_dets)} detections to {args.out}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
