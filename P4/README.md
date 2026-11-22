# P4 — Inventory builder and visualization

Turns one detection record per brick into a counted inventory, and draws those pieces back onto the input photo.

P1 (boxes) and P3 (color ids) do not have a real output format in the repo yet. The demo uses mock records in the shape those stages will hand off. Part ids match the ids P2 already predicts (`part_id_current` in `P2/classes_50.csv`).

## Input format

`build_inventory` takes a list. Each item is one brick:

| field | meaning |
| --- | --- |
| `bounding_box` | `(x_min, y_min, x_max, y_max)` in pixels on the input image |
| `part_id` | part id from the shape classifier (P2). Use a string, matching `part_id_current`. |
| `color_id` | color id from the color classifier (P3). Use an int. |

Bricks with the same `part_id` and the same `color_id` become one line: a `count`, plus `boxes` (every bounding box for that line, in the order they were given). `annotate_image(image, inventory)` draws those boxes on a copy of the photo. Pieces on the same line share one outline color.

## Run the demo

From the repository root, with the same environment that already has Pillow (P2 uses it):

```bash
jupyter notebook P4/notebooks/01_inventory_demo.ipynb
```

Open the notebook and run all cells. It builds a mock photo, prints the inventory, and shows the photo with each piece highlighted.

No new packages. Grouping is plain Python. Drawing uses Pillow, which this repo already imports in `P2/classifier.py`.
