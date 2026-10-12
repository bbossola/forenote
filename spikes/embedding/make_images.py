# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow==11.3.0"]
# ///
"""Generate the 4 fixture images for the image retrieval check."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent / "images"
SIZE = 448


def canvas():
    img = Image.new("RGB", (SIZE, SIZE), "white")
    return img, ImageDraw.Draw(img)


def main():
    OUT.mkdir(exist_ok=True)

    img, d = canvas()
    d.rectangle([100, 100, 348, 348], fill=(220, 20, 20))
    img.save(OUT / "red_square.png")

    img, d = canvas()
    d.ellipse([100, 100, 348, 348], fill=(20, 60, 220))
    img.save(OUT / "blue_circle.png")

    img, d = canvas()
    d.polygon([(224, 80), (380, 360), (68, 360)], fill=(20, 170, 50))
    img.save(OUT / "green_triangle.png")

    img, d = canvas()
    font = ImageFont.load_default(size=110)
    d.text((SIZE / 2, SIZE / 2), "HELLO", fill="black", font=font, anchor="mm")
    img.save(OUT / "hello_text.png")

    print(f"wrote 4 images to {OUT}")


if __name__ == "__main__":
    main()
