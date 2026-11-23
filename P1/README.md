# P1 - Brick Detector

Finds every LEGO brick in a photo and draws a box around it. It only answers "where are the bricks", not which part each one is (that's P2).

- **Model:** YOLOv8n, fine-tuned from the COCO-pretrained weights, all layers trained (30 epochs max, early stopping after 8 without improvement)
- **Input:** a photo, resized by YOLO to 1280 x 1280
- **Output:** an N x 6 array per photo, one row per brick: `x1, y1, x2, y2, confidence, class` (pixels of the original photo, class is always 0 = `brick`)
- **Loss:** default YOLO box and classification loss
- **Metrics:** mAP@0.5 and recall on held-out test photos

## Data

*Tagged images with LEGO bricks* (Boiński, 2021), from [mostwiedzy.pl](https://mostwiedzy.pl/en/open-research-data/tagged-images-with-lego-bricks,202309140833448152311-0).
The zip (6.0 GB) has 2,933 real photos and 2,908 renders, each with a Pascal VOC `.xml` file giving one box per brick. Photos sit in `photos/<n>/`, where `<n>` is the number of bricks in the photo.

| Bricks per photo | Photos |
|---|---|
| 1 | 2,392 |
| 2-5 | 454 |
| 6-32 | 87 |
| **Total** | **2,933 photos, 4,780 bricks** |

Only the real photos are used. The renders show one brick on a plain background, which is not what the detector sees in practice.

**Split.** 70/15/15 train/dev/test (2,053 / 441 / 439 photos, seed 539). Photos taken less than 60 seconds apart are usually the same scene shot again, so they are grouped (199 groups) and each group goes entirely into one split. This keeps near-copies of training photos out of the test set. Groups with multi-brick photos are split separately so dev and test both get busy scenes.

**Conversion to YOLO format.** VOC stores corners in pixels (`xmin ymin xmax ymax`). YOLO wants one `.txt` per image with `class cx cy w h`, all as fractions of the image size. All boxes become class 0 (the dataset calls them `lego`, plus 61 typos `legod`). Photos are shrunk to 1280 px on the long side; since labels are fractions, they don't change.

## Files

```
P1/
├── detector.py                    label conversion, scene grouping and split, detect(), evaluation helpers
└── notebooks/
    ├── 01_build_dataset.ipynb     download, convert, split, save lego_det.zip to Drive
    └── 02_train.ipynb             train, pick threshold on dev, final test evaluation
```

## How to run (Colab, GPU runtime)

1. Open `notebooks/01_build_dataset.ipynb` in Colab and run all cells. It downloads the dataset and saves `MyDrive/ece539-p1/data/lego_det.zip` and `MyDrive/ece539-p1/results/splits.csv`. If the download is blocked, upload `detector_data.zip` to `MyDrive/ece539-p1/raw/` and re-run.
2. Open `notebooks/02_train.ipynb`, set the runtime to GPU (T4), and run all cells. Set `SMOKE_TEST = True` first for a quick check that everything runs.
   Training takes roughly 30-45 minutes on a T4 and prints the time left after every epoch. YOLOv8n is used because it is the fastest YOLOv8; `yolov8s.pt` is about 2x slower and slightly more accurate.
   If Colab disconnects, re-run from the top. Training resumes from the last checkpoint on Drive.
3. Results are saved to `MyDrive/ece539-p1/results/runs/<run>/` (`test_metrics.json`, `dev_thresholds.csv`, `curves.png`, example predictions). Weights are saved to `MyDrive/ece539-p1/checkpoints/<run>/weights/best.pt`.

## Using the detector

```python
import detector

model = detector.load_model("best.pt")
boxes = detector.detect(model, "photo.jpg", conf=0.5)   # use the threshold from test_metrics.json
# boxes: N x 6 numpy array, x1 y1 x2 y2 confidence class
```

## Results

To be filled in after the full training run.

| Run | Test mAP@0.5 | Test recall | Precision | Count accuracy | Threshold |
|---|---|---|---|---|---|
| yolov8n_1280 | | | | | |
