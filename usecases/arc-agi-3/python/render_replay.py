"""
render_replay.py — render a short LS20 episode as a PNG image grid.

Usage:
    python render_replay.py [--steps N] [--out replay.png] [--scale S]

Replays the canonical 20-step action sequence and saves every Nth frame
side by side so the user can see what the game actually looks like.
"""
import sys, argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parents[2]))

import arc_agi
from agents import obs_frame

# Exact SDK color map (from arc_agi/rendering.py)
COLOR_MAP: dict[int, tuple[int, int, int]] = {
    0:  (0xFF, 0xFF, 0xFF),  # White
    1:  (0xCC, 0xCC, 0xCC),  # Off-white
    2:  (0x99, 0x99, 0x99),  # neutral Light
    3:  (0x66, 0x66, 0x66),  # neutral (background green in LS20)
    4:  (0x33, 0x33, 0x33),  # Off Black
    5:  (0x00, 0x00, 0x00),  # Black
    6:  (0xE5, 0x3A, 0xA3),  # Magenta
    7:  (0xFF, 0x7B, 0xCC),  # Magenta Light
    8:  (0xF9, 0x3C, 0x31),  # Red
    9:  (0x1E, 0x93, 0xFF),  # Blue
    10: (0x88, 0xD8, 0xF1),  # Blue Light
    11: (0xFF, 0xDC, 0x00),  # Yellow
    12: (0xFF, 0x85, 0x1B),  # Orange
    13: (0x92, 0x12, 0x31),  # Maroon
    14: (0x4F, 0xCC, 0x30),  # Green
    15: (0xA3, 0x56, 0xD6),  # Purple
}
FALLBACK = (0x80, 0x80, 0x80)


def frame_to_image(frame: list, scale: int = 6) -> Image.Image:
    h, w = len(frame), len(frame[0])
    img = Image.new("RGB", (w * scale, h * scale))
    pixels = img.load()
    for r, row in enumerate(frame):
        for c, val in enumerate(row):
            rgb = COLOR_MAP.get(val, FALLBACK)
            for dr in range(scale):
                for dc in range(scale):
                    pixels[c * scale + dc, r * scale + dr] = rgb
    return img


def add_label(img: Image.Image, text: str) -> Image.Image:
    bar_h = 18
    new = Image.new("RGB", (img.width, img.height + bar_h), (30, 30, 30))
    new.paste(img, (0, bar_h))
    draw = ImageDraw.Draw(new)
    draw.text((4, 2), text, fill=(220, 220, 220))
    return new


def make_grid(images: list[Image.Image], cols: int) -> Image.Image:
    rows = (len(images) + cols - 1) // cols
    w, h = images[0].width, images[0].height
    grid = Image.new("RGB", (w * cols, h * rows), (10, 10, 10))
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        grid.paste(im, (c * w, r * h))
    return grid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--out",   default="replay.png")
    p.add_argument("--scale", type=int, default=6)
    p.add_argument("--cols",  type=int, default=5)
    args = p.parse_args()

    action_map = {1: "ACTION1", 2: "ACTION2", 3: "ACTION3", 4: "ACTION4"}
    # Canonical 20-step sequence from prior test runs
    actions = [1,2,3,4, 1,1,3,3, 2,1,4,1,1, 1,1,1,1,1, 3,2][:args.steps]

    arc = arc_agi.Arcade()
    env = arc.make("ls20")
    obs = env.reset()

    frames = []
    frame = obs_frame(obs)
    frames.append((0, "reset", frame))

    for i, act in enumerate(actions, 1):
        result = env.step(act)
        frame = obs_frame(result)
        frames.append((i, action_map[act], frame))

    images = []
    for step, label, f in frames:
        img = frame_to_image(f, scale=args.scale)
        img = add_label(img, f"step {step}: {label}")
        images.append(img)

    grid = make_grid(images, cols=args.cols)
    out = Path(__file__).parent / args.out
    grid.save(out)
    print(f"Saved {len(images)} frames -> {out}  ({grid.width}x{grid.height}px)")


if __name__ == "__main__":
    main()
