import random
import re
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw

CLASS_NAMES = ["brick"]
IMAGE_SIZE = 1280

# The dataset labels every box "lego", except for a few typos ("legod").
# The detector only has one class, so every box becomes class 0.
BRICK = 0


def read_voc(xml_bytes):
    # Returns (width, height, boxes) where boxes is a list of (xmin, ymin, xmax, ymax) in pixels
    root = ET.fromstring(xml_bytes)
    width = int(root.findtext("size/width"))
    height = int(root.findtext("size/height"))
    boxes = []
    for obj in root.findall("object"):
        b = obj.find("bndbox")
        boxes.append(tuple(float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")))
    return width, height, boxes


def to_yolo_lines(boxes, width, height):
    # YOLO label format: one line per box, "class cx cy w h", all scaled to 0-1
    lines = []
    for x1, y1, x2, y2 in boxes:
        cx = (x1 + x2) / 2 / width
        cy = (y1 + y2) / 2 / height
        w = (x2 - x1) / width
        h = (y2 - y1) / height
        lines.append(f"{BRICK} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def read_yolo_labels(path, width, height):
    # Reverse of to_yolo_lines: returns an N x 4 array of (x1, y1, x2, y2) in pixels
    rows = [line.split() for line in open(path).read().splitlines() if line.strip()]
    boxes = np.array([[float(v) for v in r[1:5]] for r in rows]).reshape(-1, 4)
    cx, cy, w, h = boxes[:, 0] * width, boxes[:, 1] * height, boxes[:, 2] * width, boxes[:, 3] * height
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def photo_time(stem):
    # The photos come from three sources, each with its own naming scheme:
    #   IMG_20201211_171151                  phone photo, date and time in the name
    #   0_EnU9_original_1608917990980        sorter app, millisecond timestamp at the end
    #   3022_flash_07 / 3022_no_flash_02     part number, then shot number
    # Returns ("phone" | "app", seconds) or ("part", part number)
    m = re.match(r"IMG_(\d{8}_\d{6})", stem)
    if m:
        return "phone", datetime.strptime(m[1], "%Y%m%d_%H%M%S").timestamp()
    m = re.search(r"_(\d{13})$", stem)
    if m:
        return "app", int(m[1]) / 1000
    m = re.match(r"(\w+?)_(?:no_)?flash_\d+$", stem)
    if m:
        return "part", m[1]
    raise ValueError(f"Unrecognised photo name: {stem}")


def group_photos(stems, gap_seconds=60):
    # Photos taken within gap_seconds of each other are usually the same scene shot a few times.
    # They get the same group id so a split never puts near-duplicates in both train and test.
    # Photos of the same part number (the flash / no_flash set) are grouped together too.
    order = sorted(stems, key=lambda s: (photo_time(s), s))
    groups = {}
    group_id, last = -1, None
    for stem in order:
        source, t = photo_time(stem)
        if source == "part":
            same = last is not None and last[0] == "part" and last[1] == t
        else:
            same = last is not None and last[0] == source and t - last[1] <= gap_seconds
        if not same:
            group_id += 1
        groups[stem] = group_id
        last = (source, t)
    return groups


def split_groups(group_sizes, fractions, seed):
    # group_sizes: {group_id: number of photos}. Returns {group_id: "train" | "dev" | "test"}.
    # Whole groups are handed out, biggest first, each to whichever split is furthest below its target.
    names = ("train", "dev", "test")
    total = sum(group_sizes.values())
    target = {name: f * total for name, f in zip(names, fractions)}
    filled = {name: 0 for name in names}

    ids = sorted(group_sizes)
    random.Random(seed).shuffle(ids)
    ids.sort(key=lambda g: -group_sizes[g])  # stable sort, so ties stay in shuffled order

    assignment = {}
    for g in ids:
        name = max(names, key=lambda n: target[n] - filled[n])
        assignment[g] = name
        filled[name] += group_sizes[g]
    return assignment


def resize_max_side(img, max_side):
    # Shrinks the photo so its longest side is max_side. Boxes are stored as fractions so they still line up.
    scale = max_side / max(img.size)
    if scale >= 1:
        return img
    new_size = (round(img.width * scale), round(img.height * scale))
    return img.resize(new_size, Image.BILINEAR)


def data_yaml(root):
    # The dataset description file ultralytics reads before training
    return (f"path: {root}\n"
            "train: images/train\n"
            "val: images/dev\n"
            "test: images/test\n"
            f"names:\n  0: {CLASS_NAMES[0]}\n")


def load_model(weights_path):
    from ultralytics import YOLO
    return YOLO(str(weights_path))


def detect(model, images, conf=0.25, iou=0.7, imgsz=IMAGE_SIZE, device=None):
    # images: a single image or a list (file paths, PIL images or numpy arrays)
    # Returns one N x 6 array per image: x1, y1, x2, y2, confidence, class
    # Box coordinates are in pixels of the original image, not the resized one.
    single = not isinstance(images, list)
    results = model.predict(images if not single else [images], conf=conf, iou=iou, imgsz=imgsz,
                            device=device, verbose=False)
    out = []
    for r in results:
        b = r.boxes
        out.append(np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None],
                                   b.cls.cpu().numpy()[:, None]], axis=1).astype(np.float32))
    return out[0] if single else out


def box_iou(a, b):
    # IoU between every box in a (N x 4) and every box in b (M x 4), result is N x M
    a, b = np.asarray(a, dtype=float).reshape(-1, 4), np.asarray(b, dtype=float).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter)


def match(dets, gt_boxes, iou_thr=0.5):
    # Marks each detection as a true positive or not. Highest confidence detections get first pick
    # of the ground truth boxes, and each ground truth box can only be matched once.
    # Returns (confidences, is_true_positive), both sorted by confidence from high to low.
    if len(dets) == 0:
        return np.zeros(0), np.zeros(0, dtype=bool)
    dets = dets[np.argsort(-dets[:, 4])]
    hit = np.zeros(len(dets), dtype=bool)
    if len(gt_boxes):
        ious = box_iou(dets[:, :4], gt_boxes)
        taken = np.zeros(len(gt_boxes), dtype=bool)
        for i in range(len(dets)):
            candidates = np.where(~taken & (ious[i] >= iou_thr))[0]
            if len(candidates):
                j = candidates[np.argmax(ious[i, candidates])]
                taken[j] = True
                hit[i] = True
    return dets[:, 4], hit


def threshold_table(per_image, thresholds, iou_thr=0.5):
    # per_image: list of (dets, gt_boxes). For each confidence threshold works out precision, recall, F1,
    # and how often the number of detected bricks in a photo exactly equals the true number.
    # The matching is done once at the lowest threshold; raising the threshold only drops low confidence
    # detections, which never change what the higher confidence ones matched.
    matched = [match(d, g, iou_thr) for d, g in per_image]
    n_gt = sum(len(g) for _, g in per_image)
    rows = []
    for t in thresholds:
        tp = fp = exact = 0
        for (conf, hit), (_, gt) in zip(matched, per_image):
            keep = conf >= t
            tp += int(hit[keep].sum())
            fp += int((~hit[keep]).sum())
            exact += int(keep.sum() == len(gt))
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / n_gt if n_gt else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"conf": float(t), "precision": precision, "recall": recall, "f1": f1,
                     "count_accuracy": exact / len(per_image), "tp": tp, "fp": fp, "fn": n_gt - tp})
    return rows


def draw_boxes(img, dets, gt_boxes=None):
    # Green = ground truth, red = detections with their confidence
    img = img.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    width = max(2, round(max(img.size) / 400))
    for box in (gt_boxes if gt_boxes is not None else []):
        draw.rectangle(list(box), outline=(0, 200, 0), width=width)
    for x1, y1, x2, y2, conf, _ in dets:
        draw.rectangle([x1, y1, x2, y2], outline=(230, 0, 0), width=width)
        draw.text((x1 + width, y1 + width), f"{conf:.2f}", fill=(230, 0, 0))
    return img
