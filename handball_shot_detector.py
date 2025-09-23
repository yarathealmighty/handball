
#!/usr/bin/env python3
\"\"\"handball_shot_detector.py

Usage:
    python3 handball_shot_detector.py --video path/to/match.mp4 --model path/to/yolov12_weights.pt

What it does (best-effort):
- Loads a detector (YOLOv12 or any detector implementing a small wrapper)
- Lets you click 2 points to define a "shot line" near each goal (left and right) on a single frame
- Optionally click 2 points to define the actual "goal line" for each goal
- Runs detection + simple centroid tracking for the ball
- Registers a 'shot' when the ball centroid crosses the shot-line into the goal half
- Marks it as a 'goal' if within a short window after the shot the ball crosses the goal-line
- Saves results to CSV with timestamp (seconds), frame index, and label ('goal'/'miss')

NOTES:
- You must provide a working detector function or a YOLOv12 weights file compatible with the load_yolov12_model() function below.
- The script is intentionally modular: replace the Detector.detect() method if you use a different model or API.
- Designed for inference-only. For best results run on GPU-enabled PyTorch environment (for AMD GPU, use ROCm-enabled PyTorch).
\"\"\"

import argparse
import csv
import math
import time
from collections import deque

import cv2
import numpy as np

try:
    import torch
except Exception:
    torch = None
    print(\"WARNING: PyTorch not available. You must install torch and a yolov12-compatible model to use the YOLO detector.\")

# --------------------- Detector wrapper ---------------------
class Detector:
    \"\"\"Wrap your model here. The detect(frame) method must return a list of detections:
    detections = [{'bbox': [x1,y1,x2,y2], 'label': 'ball', 'score': 0.9}, ...]
    \"\"\"
    def __init__(self, model_path=None, device='cpu', conf_thres=0.25):
        self.model_path = model_path
        self.device = device
        self.conf_thres = conf_thres
        self.model = None
        if model_path is not None:
            self.model = self.load_yolov12_model(model_path, device)
        else:
            print(\"Detector initialized without model. Falling back to simple blob detector (toy/demo only). See script comments.\")
            self.blob = self._create_blob_detector()

    def load_yolov12_model(self, model_path, device='cpu'):
        \"\"\"Attempt to load YOLOv12 PyTorch model.
        Replace or extend this loader if you use a different repo or ONNX runtime.
        \"\"\"
        if torch is None:
            raise RuntimeError(\"PyTorch not found. Install torch to load model.\")
        # -- Simple generic loader: try torch.jit, then torch.load
        try:
            model = torch.jit.load(model_path, map_location=device)
            model.to(device)
            model.eval()
            print(f\"Loaded TorchScript model from {model_path}\")
            return model
        except Exception as e:
            print(f\"TorchScript load failed: {e}. Trying torch.load(...)\")
        try:
            model = torch.load(model_path, map_location=device)
            if hasattr(model, 'eval'):
                model.eval()
            print(f\"Loaded stateful model from {model_path}\")
            return model
        except Exception as e:
            raise RuntimeError(f\"Failed to load model: {e}\\nYou need to supply a YOLOv12-compatible model or adapt the loader.\")

    def _create_blob_detector(self):
        # Create a simple blob detector tuned for ball-like blobs. This is a fallback for quick testing only.
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 50
        params.maxArea = 10000
        params.filterByCircularity = False
        params.filterByInertia = False
        params.filterByConvexity = False
        return cv2.SimpleBlobDetector_create(params)

    def detect(self, frame):
        if self.model is None:
            # fallback: convert to grayscale and detect blobs
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            keypoints = self.blob.detect(gray)
            dets = []
            for kp in keypoints:
                x = int(kp.pt[0] - kp.size/2)
                y = int(kp.pt[1] - kp.size/2)
                s = kp.size
                dets.append({'bbox': [x, y, int(x+s), int(y+s)], 'label': 'ball', 'score': 0.5})
            return dets

        # Example generic inference stub for YOLO-like models that accept numpy images.
        # You may need to adapt this block depending on how your model expects input.
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_t = torch.from_numpy(img).permute(2,0,1).float().unsqueeze(0) / 255.0
        img_t = img_t.to(self.device)
        with torch.no_grad():
            try:
                preds = self.model(img_t)
            except Exception as e:
                raise RuntimeError(f\"Model inference failed: {e}\\nAdapt Detector.detect() to your model API.\")
        # Expect preds to be a list/tuple with boxes, scores, labels (this is repo-dependent)
        # Try to handle a few common output formats:
        dets = []
        # Case: model returns a tensor Nx6 like (x1,y1,x2,y2,score,class)
        if isinstance(preds, (list, tuple)):
            out = preds[0]
        else:
            out = preds
        if hasattr(out, 'cpu'):
            out = out.cpu().numpy()
        # If shape is (N,6)
        if isinstance(out, np.ndarray) and out.ndim == 2 and out.shape[1] >= 6:
            for r in out:
                x1,y1,x2,y2,score,cls = r[:6]
                if score < self.conf_thres: 
                    continue
                dets.append({'bbox': [int(x1),int(y1),int(x2),int(y2)], 'label': str(int(cls)), 'score': float(score)})
            return dets
        # Otherwise, user should extend this to match model outputs.
        raise RuntimeError('Unexpected model output format. Please adapt Detector.detect() for your weights.')

# --------------------- Utilities ---------------------
def draw_lines(img, shot_lines, goal_lines):
    for ln in shot_lines:
        cv2.line(img, tuple(ln[0]), tuple(ln[1]), (0,255,255), 2)
    for ln in goal_lines:
        cv2.line(img, tuple(ln[0]), tuple(ln[1]), (0,0,255), 2)

def point_side_of_line(pt, a, b):
    # returns signed distance from line ab to point pt (positive on one side, negative on other)
    x0,y0 = pt
    x1,y1 = a
    x2,y2 = b
    return (x2-x1)*(y0-y1) - (y2-y1)*(x0-x1)

# Simple tracker that tracks the ball centroid across frames by nearest match
class SimpleCentroidTracker:
    def __init__(self, max_lost=10):
        self.next_id = 0
        self.objects = {}  # id -> (centroid, last_seen_frame)
        self.max_lost = max_lost

    def update(self, detections, frame_idx):
        # detections: list of centroids
        centers = [((d['bbox'][0]+d['bbox'][2])//2, (d['bbox'][1]+d['bbox'][3])//2) for d in detections]
        assigned = {}
        results = {}
        # match centers to existing objects by closest Euclidean distance
        for cid, center in enumerate(centers):
            best_id = None
            best_dist = float('inf')
            for oid, (ocenter, last_seen) in list(self.objects.items()):
                dist = math.hypot(center[0]-ocenter[0], center[1]-ocenter[1])
                if dist < best_dist and dist < 80:  # threshold for matching
                    best_dist = dist
                    best_id = oid
            if best_id is not None:
                # assign
                self.objects[best_id] = (center, frame_idx)
                results[best_id] = center
            else:
                # create new
                oid = self.next_id
                self.next_id += 1
                self.objects[oid] = (center, frame_idx)
                results[oid] = center
        # remove stale
        stale = [oid for oid,(c,last_seen) in self.objects.items() if frame_idx - last_seen > self.max_lost]
        for s in stale:
            del self.objects[s]
        return results

# --------------------- Interactive helpers ---------------------
class LineSelector:
    def __init__(self, frame):
        self.frame = frame.copy()
        self.points = []
        self.window = 'select-lines (press q when done)'
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, 1200, 700)
        cv2.setMouseCallback(self.window, self.on_mouse)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                self.points.append((x,y))
                cv2.circle(self.frame, (x,y), 4, (0,255,0), -1)
                if len(self.points) % 2 == 0:
                    a = self.points[-2]; b = self.points[-1]
                    cv2.line(self.frame, a, b, (255,255,0), 2)
            else:
                print('Already selected 4 points (two lines). Press q to finish.')

    def run(self):
        print('Click 2 points for shot-line of goal A, then 2 points for shot-line of goal B')
        while True:
            disp = self.frame.copy()
            cv2.imshow(self.window, disp)
            k = cv2.waitKey(10) & 0xFF
            if k == ord('q') or len(self.points) >= 4:
                break
        cv2.destroyWindow(self.window)
        if len(self.points) < 4:
            raise RuntimeError('Not enough points selected.')
        return [(self.points[0], self.points[1]), (self.points[2], self.points[3])]

# --------------------- Main pipeline ---------------------
def process_video(video_path, detector, shot_lines, goal_lines=None, conf_label='ball', out_csv='shots.csv', run_every_n=2, goal_window_s=2.0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {video_path}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f'Video FPS: {fps:.2f}, frames: {total_frames}')

    tracker = SimpleCentroidTracker(max_lost=15)
    shots = []  # list of dicts: {'frame':, 'time':, 'goal':True/False, 'ball_pos':(x,y), 'side':0/1}

    last_ball_positions = deque(maxlen=30)  # recent centroids for smoothing
    frame_idx = 0

    # We'll detect every run_every_n frames to save compute; track in between using last known position
    last_detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        process_frame = (frame_idx % run_every_n == 0)

        detections = []
        if process_frame:
            dets = detector.detect(frame)
            # keep only those labeled as ball (or first detection if detector doesn't label)
            if dets:
                # pick the detection with largest area if multiple candidates
                dets_ball = [d for d in dets if d.get('label','').lower() in ('ball','0','soccer','football','handball')]
                if len(dets_ball) == 0:
                    dets_ball = dets  # fallback accept any detection
                # sort by area descending
                dets_ball.sort(key=lambda d: (d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]), reverse=True)
                detections = dets_ball[:1]  # choose top candidate
            else:
                detections = []
            last_detections = detections
        else:
            detections = last_detections

        tracked = tracker.update(detections, frame_idx)
        # choose one ball object to follow: the one with smallest id (stable)
        if len(tracked) > 0:
            oid = sorted(tracked.keys())[0]
            ball_pos = tracked[oid]
            last_ball_positions.append(ball_pos)
            ball_centroid = (int(sum([p[0] for p in last_ball_positions])/len(last_ball_positions)),
                             int(sum([p[1] for p in last_ball_positions])/len(last_ball_positions)))
        else:
            ball_centroid = None

        # Shot detection: check if centroid crosses shot_line from outside->inside
        if ball_centroid is not None:
            for side_idx, shot_line in enumerate(shot_lines):
                a,b = shot_line
                side = point_side_of_line(ball_centroid, a, b)
                # We need the previous side value to know if it crossed
                if 'prev_side_{}'.format(side_idx) not in globals():
                    globals()['prev_side_{}'.format(side_idx)] = None
                prev_side = globals().get('prev_side_{}'.format(side_idx))
                if prev_side is not None and prev_side < 0 and side >= 0:
                    # crossing from negative to positive: interpreted as entering goal half (adjust sign to your coordinate selection)
                    # register shot
                    t = frame_idx / fps
                    print(f'[SHOT] frame {frame_idx} time {t:.2f}s side {side_idx} pos {ball_centroid}')
                    shots.append({'frame': frame_idx, 'time': t, 'goal': False, 'side': side_idx, 'ball_pos': ball_centroid, 'checked_goal': False})
                globals()['prev_side_{}'.format(side_idx)] = side

        # check for goals: if ball crosses goal_line within goal_window_s after a shot
        for s in shots:
            if s['checked_goal']:
                continue
            if frame_idx < s['frame']:
                continue
            if (frame_idx - s['frame'])/fps > goal_window_s:
                s['checked_goal'] = True  # window expired
                continue
            # check crossing of goal_line for the same side
            if ball_centroid is None:
                continue
            goal_line = goal_lines[s['side']] if goal_lines else shot_lines[s['side']]
            a,b = goal_line
            prev_goal_side = globals().get('prev_goal_side_{}_{}'.format(s['side'], id(s)), None)
            cur_goal_side = point_side_of_line(ball_centroid, a, b)
            # store prev for next frame
            globals()['prev_goal_side_{}_{}'.format(s['side'], id(s))] = cur_goal_side
            if prev_goal_side is not None and prev_goal_side < 0 and cur_goal_side >= 0:
                # ball crossed the goal line -> goal
                s['goal'] = True
                s['checked_goal'] = True
                print(f'[GOAL] shot at {s[\"time\"]:.2f}s confirmed as GOAL')

        # Optional: visualization while running (comment out to speed up)
        vis = frame.copy()
        draw_lines(vis, shot_lines, goal_lines if goal_lines else [])
        if ball_centroid is not None:
            cv2.circle(vis, ball_centroid, 6, (0,255,0), -1)
        cv2.putText(vis, f'Frame {frame_idx}/{total_frames}', (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imshow('processing', vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # finalize: mark remaining un-checked shots as misses
    for s in shots:
        if not s['checked_goal']:
            s['checked_goal'] = True

    # Save CSV
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['frame','time','side','goal','ball_pos'])
        writer.writeheader()
        for s in shots:
            writer.writerow({'frame': s['frame'], 'time': f\"{s['time']:.3f}\", 'side': s['side'], 'goal': int(s['goal']), 'ball_pos': s['ball_pos']})
    print(f'Saved {len(shots)} shots to {out_csv}')
    for s in shots:
        print(f\"{s['time']:.2f}s - {'GOAL' if s['goal'] else 'MISS'} (frame {s['frame']})\")
    return shots

# --------------------- CLI ---------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True, help='path to match video')
    parser.add_argument('--model', required=False, default=None, help='path to model weights (optional)')
    parser.add_argument('--out', default='shots.csv', help='output CSV file')
    parser.add_argument('--device', default='cpu', help='torch device (cpu or cuda or mps or rocm device)')
    parser.add_argument('--run-every-n', type=int, default=2, help='run detection every N frames to speed up')
    parser.add_argument('--goal-window', type=float, default=2.0, help='seconds after shot to consider for goal detection')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError('Cannot open video to select lines.')
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError('Cannot read frame from video.')

    selector = LineSelector(frame)
    shot_lines = selector.run()
    # Let user optionally select goal-lines; here we reuse LineSelector for goal lines, but it's optional.
    print('Now optionally select goal-lines (2 points per goal). If you prefer, press q to skip and shot-lines will be used as goal-lines.')
    selector2 = LineSelector(frame)
    try:
        goal_lines = selector2.run()
    except Exception as e:
        print('No goal-lines selected, using shot-lines as goal-lines.')
        goal_lines = shot_lines

    detector = Detector(model_path=args.model, device=args.device)
    shots = process_video(args.video, detector, shot_lines, goal_lines, out_csv=args.out, run_every_n=args.run_every_n, goal_window_s=args.goal_window)

if __name__ == '__main__':
    main()
