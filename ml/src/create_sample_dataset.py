import argparse
import csv
import os
import random

from PIL import Image, ImageDraw


def draw_scene(size=256):
    bg_options = ["skyblue", "lightgreen", "lightyellow", "lightgray", "lavender"]
    objects = ["red ball", "blue square", "yellow triangle"]

    image = Image.new("RGB", (size, size), random.choice(bg_options))
    draw = ImageDraw.Draw(image)

    used_objects = []
    count = random.randint(1, 3)
    picks = random.sample(objects, count)

    for obj in picks:
        x1 = random.randint(20, size - 120)
        y1 = random.randint(20, size - 120)
        x2 = x1 + random.randint(40, 90)
        y2 = y1 + random.randint(40, 90)

        if obj == "red ball":
            draw.ellipse([x1, y1, x2, y2], fill="red", outline="black")
        elif obj == "blue square":
            draw.rectangle([x1, y1, x2, y2], fill="blue", outline="black")
        elif obj == "yellow triangle":
            draw.polygon([(x1, y2), ((x1 + x2) // 2, y1), (x2, y2)], fill="yellow", outline="black")
        used_objects.append(obj)

    caption = "a scene with " + " and ".join(used_objects)
    return image, caption


def generate(output_dir, csv_path, samples):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    rows = []
    for i in range(samples):
        img, caption = draw_scene()
        img_path = os.path.join(output_dir, f"sample_{i:03d}.png")
        img.save(img_path)
        rows.append([img_path.replace("\\", "/"), caption])

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "caption"])
        writer.writerows(rows)

    print(f"Saved {samples} images to: {output_dir}")
    print(f"Saved captions CSV to: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/sample_images")
    parser.add_argument("--csv-path", default="data/captions.csv")
    parser.add_argument("--samples", type=int, default=120)
    args = parser.parse_args()
    generate(args.output_dir, args.csv_path, args.samples)
