from pathlib import Path
import shutil

SOURCE = Path.home() / "Downloads" / "idcard_dataset"
DEST = Path.home() / "Downloads" / "idcard_clean"

splits = ["train", "valid", "test"]


def polygon_to_box(values):
    # values = x1 y1 x2 y2 x3 y3 ...
    xs = values[0::2]
    ys = values[1::2]

    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)

    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    width = xmax - xmin
    height = ymax - ymin

    return x_center, y_center, width, height


for split in splits:
    image_dir = SOURCE / split / "images"
    label_dir = SOURCE / split / "labels"

    dest_image_dir = DEST / split / "images"
    dest_label_dir = DEST / split / "labels"

    dest_image_dir.mkdir(parents=True, exist_ok=True)
    dest_label_dir.mkdir(parents=True, exist_ok=True)

    for image in image_dir.iterdir():
        if image.is_file():
            shutil.copy2(image, dest_image_dir / image.name)

    for label in label_dir.glob("*.txt"):
        output_lines = []

        for line in label.read_text().splitlines():
            parts = line.split()

            if len(parts) < 5:
                continue

            class_id = parts[0]
            coords = list(map(float, parts[1:]))

            # Already a normal YOLO box
            if len(coords) == 4:
                box = coords

            # Polygon -> bounding box
            else:
                if len(coords) % 2 != 0:
                    print(f"Skipping malformed line in {label.name}")
                    continue

                box = polygon_to_box(coords)

            output_lines.append(
                class_id + " " + " ".join(f"{v:.6f}" for v in box)
            )

        (dest_label_dir / label.name).write_text(
            "\n".join(output_lines) + ("\n" if output_lines else "")
        )


# Create data.yaml
data_yaml = f"""path: {DEST.as_posix()}
train: train/images
val: valid/images
test: test/images

nc: 2
names: ['0', '1']
"""

(DEST / "data.yaml").write_text(data_yaml)

print("======================================")
print("DATASET CONVERSION COMPLETED")
print("======================================")
print(f"New dataset: {DEST}")
print("Original dataset was NOT modified.")