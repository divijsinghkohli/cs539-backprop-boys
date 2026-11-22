from PIL import Image, ImageDraw, ImageFont

# outline colors for each group. not the bricks real color,
#otherwise a red brick just blends into a red box
OUTLINE_COLORS = [
    (220, 30, 220),
    (0, 150, 150),
    (230, 120, 0),
    (60, 90, 220),
    (140, 50, 180),
    (0, 140, 60),
]


def _inventory_sort_key(group):
    color_id = group["color_id"]

    # ints sort as numbers so color 2 comes before 14
    if isinstance(color_id, int):
        color_key = (0, color_id)

    #anything else just sorts as text
    else:
        color_key = (1, str(color_id))

    return (str(group["part_id"]), color_key)


def _label_font():
    # newer pillow lets you pass a size
    try:
        return ImageFont.load_default(size=16)

    #older pillow doesnt, so use the tiny default
    except TypeError:
        return ImageFont.load_default()


def build_inventory(detections):
    # same part id AND same color id go on one line.
    #keep the boxes so we can higlight those pieces later

    #1. one bucket per (part id, color id)
    groups = {}

    #2. go thru the bricks one by one
    for detection in detections:

        # make sure we actually got a dict with the 3 fields
        if not isinstance(detection, dict):
            raise ValueError(
                "each detection must be a dict with bounding_box, part_id, and color_id"
            )
        if (
            "bounding_box" not in detection
            or "part_id" not in detection
            or "color_id" not in detection
        ):
            raise ValueError("each detection needs bounding_box, part_id, and color_id")

        #3. box has to be x_min, y_min, x_max, y_max
        bounding_box = detection["bounding_box"]
        if len(bounding_box) != 4:
            raise ValueError("bounding_box must be (x_min, y_min, x_max, y_max)")

        # pull out the ids. both have to match to land in the same group
        part_id = detection["part_id"]
        color_id = detection["color_id"]
        group_key = (part_id, color_id)

        #4. first time we see this pair, start a new line
        if group_key not in groups:
            groups[group_key] = {
                "part_id": part_id,
                "color_id": color_id,
                "count": 0,
                "boxes": [],
            }

        # add one to the count and hang onto the box
        groups[group_key]["count"] = groups[group_key]["count"] + 1
        groups[group_key]["boxes"].append(bounding_box)

    #5. list is easier to print. sort so the same input doesnt jump around
    inventory = []
    for group in groups.values():
        inventory.append(group)
    inventory.sort(key=_inventory_sort_key)

    return inventory


def annotate_image(image, inventory):
    # same inventory line gets the same outline color.
    #leave the original image alone

    #1. draw on a copy, in RGB, so we dont mess up the photo
    annotated_image = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated_image)
    font = _label_font()

    #2. walk the lines and give each one a color
    group_number = 0
    for group in inventory:
        outline_color = OUTLINE_COLORS[group_number % len(OUTLINE_COLORS)]
        group_number = group_number + 1

        # label is the part and the color, same text on every box in the line
        label = str(group["part_id"]) + "  color " + str(group["color_id"])

        #3. mark every piece that belongs to this line
        for bounding_box in group["boxes"]:
            x_min, y_min, x_max, y_max = bounding_box
            draw.rectangle([x_min, y_min, x_max, y_max],outline=outline_color,width=4,)

            #see how big the label is
            text_left, text_top, text_right, text_bottom = draw.textbbox((0, 0), label, font=font)
            text_width = text_right - text_left
            text_height = text_bottom - text_top

            # sit it just above the box. if itd get cut off, stick it inside
            label_x = x_min
            label_y = y_min - text_height - 6
            if label_y < 0:
                label_y = y_min + 2

            #colored box behind the words so you can actually read them
            draw.rectangle(
                [
                    label_x,
                    label_y,
                    label_x + text_width + 6,
                    label_y + text_height + 4,
                ],
                fill=outline_color,
            )
            draw.text((label_x + 3, label_y + 1), label,fill=(255, 255, 255),font=font)

    return annotated_image
