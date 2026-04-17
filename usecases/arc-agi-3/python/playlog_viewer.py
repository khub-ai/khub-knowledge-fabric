"""
playlog_viewer.py — interactive step-by-step viewer for arc-agi-3 playlog JSON files.

Usage:
    python playlog_viewer.py [playlog_dir]  [--level N]

Keyboard:
    Right / D / Space  — next step
    Left  / A          — prev step
    Home               — first step
    End                — last step
    Q / Escape         — quit
"""
import json
import os
import sys
import tkinter as tk
from pathlib import Path
from PIL import Image, ImageTk, ImageDraw

COLOR_MAP: dict[int, tuple[int,int,int]] = {
    0:  (0xFF,0xFF,0xFF),
    1:  (0xCC,0xCC,0xCC),
    2:  (0x99,0x99,0x99),
    3:  (0x66,0x66,0x66),
    4:  (0x33,0x33,0x33),
    5:  (0x00,0x00,0x00),
    6:  (0xE5,0x3A,0xA3),
    7:  (0xFF,0x7B,0xCC),
    8:  (0xF9,0x3C,0x31),
    9:  (0x1E,0x93,0xFF),
   10:  (0x88,0xD8,0xF1),
   11:  (0xFF,0xDC,0x00),
   12:  (0xFF,0x85,0x1B),
   13:  (0x92,0x12,0x31),
   14:  (0x4F,0xCC,0x30),
   15:  (0xA3,0x56,0xD6),
}
FALLBACK = (0x80,0x80,0x80)
SCALE = 8


def frame_to_image(frame: list, scale: int = SCALE) -> Image.Image:
    h, w = len(frame), len(frame[0])
    img = Image.new("RGB", (w*scale, h*scale))
    pixels = img.load()
    for r, row in enumerate(frame):
        for c, val in enumerate(row):
            rgb = COLOR_MAP.get(val, FALLBACK)
            for dr in range(scale):
                for dc in range(scale):
                    pixels[c*scale+dc, r*scale+dr] = rgb
    return img


def load_playlog(folder: str, level_filter: int | None = None):
    """Load all playlog JSON entries that have a frame, optionally filtered by level."""
    entries = []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(folder, fname)
        try:
            d = json.load(open(path))
        except Exception:
            continue
        ret = d.get("returned", {})
        frame = ret.get("frame")
        if frame is None:
            continue
        lvl = d.get("levels_completed", -1)
        if level_filter is not None and lvl != level_filter:
            continue
        step   = d.get("step_number", -1)
        action = d.get("action_name", "?")
        diff   = ret.get("diff", 0)
        lvl_after = ret.get("levels_completed", lvl)
        obs    = d.get("observation_state", "")
        snap   = d.get("state_snapshot", {}) or {}
        pp     = snap.get("player_position") or ret.get("player_pos")
        entries.append({
            "step": step, "action": action, "frame": frame,
            "diff": diff, "lvl": lvl, "lvl_after": lvl_after,
            "obs": obs, "pp": pp, "fname": fname,
        })
    return entries


class PlaylogViewer:
    def __init__(self, entries: list):
        self.entries = entries
        self.idx = 0

        self.root = tk.Tk()
        self.root.title("Playlog Viewer")
        self.root.configure(bg="#1e1e1e")

        # Info label
        self.info_var = tk.StringVar()
        info_lbl = tk.Label(self.root, textvariable=self.info_var,
                            font=("Consolas", 12), fg="#d4d4d4", bg="#1e1e1e",
                            justify=tk.LEFT, anchor="w")
        info_lbl.pack(fill=tk.X, padx=8, pady=(6,0))

        # Canvas
        frame_w = 64 * SCALE
        frame_h = 64 * SCALE
        self.canvas = tk.Canvas(self.root, width=frame_w, height=frame_h,
                                bg="#000000", highlightthickness=0)
        self.canvas.pack(padx=8, pady=4)

        # Navigation buttons
        btn_frame = tk.Frame(self.root, bg="#1e1e1e")
        btn_frame.pack(pady=(0,6))
        style = dict(font=("Consolas",11), bg="#333", fg="white",
                     activebackground="#555", padx=10, pady=4)
        tk.Button(btn_frame, text="|<< First", command=self.go_first, **style).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="< Prev",   command=self.go_prev,  **style).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="Next >",   command=self.go_next,  **style).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="Last >>|", command=self.go_last,  **style).pack(side=tk.LEFT, padx=4)

        # Keyboard bindings
        self.root.bind("<Right>",  lambda e: self.go_next())
        self.root.bind("<space>",  lambda e: self.go_next())
        self.root.bind("d",        lambda e: self.go_next())
        self.root.bind("<Left>",   lambda e: self.go_prev())
        self.root.bind("a",        lambda e: self.go_prev())
        self.root.bind("<Home>",   lambda e: self.go_first())
        self.root.bind("<End>",    lambda e: self.go_last())
        self.root.bind("q",        lambda e: self.root.quit())
        self.root.bind("<Escape>", lambda e: self.root.quit())

        self._photo = None
        self.render()
        self.root.mainloop()

    def render(self):
        e = self.entries[self.idx]
        img = frame_to_image(e["frame"])

        # Highlight player position if available
        pp = e.get("pp")
        if pp:
            if isinstance(pp, (list, tuple)) and len(pp) >= 2:
                cx, cy = int(pp[0]) * SCALE, int(pp[1]) * SCALE
                draw = ImageDraw.Draw(img)
                draw.rectangle([cx-2, cy-2, cx+SCALE+2, cy+SCALE+2],
                               outline=(255,0,0), width=2)

        self._photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)

        # Flag interesting steps
        flags = []
        diff = e.get("diff", 0)
        if diff >= 3000:
            flags.append("⚡ GAME RESET (life lost)")
        elif diff > 80:
            flags.append(f"★ LARGE DIFF ({diff})")
        if e.get("lvl_after", e["lvl"]) != e["lvl"]:
            flags.append(f"🏆 LEVEL UP → {e['lvl_after']}")

        flag_str = "  " + " | ".join(flags) if flags else ""
        info = (f"[{self.idx+1}/{len(self.entries)}]  "
                f"Step {e['step']}  {e['action']}  "
                f"L={e['lvl']}  diff={diff}{flag_str}")
        if e.get("pp"):
            info += f"\n  player_pos={e['pp']}"
        self.info_var.set(info)
        self.root.title(f"Step {e['step']} — {e['action']} — Playlog Viewer")

    def go_next(self):
        if self.idx < len(self.entries) - 1:
            self.idx += 1
            self.render()

    def go_prev(self):
        if self.idx > 0:
            self.idx -= 1
            self.render()

    def go_first(self):
        self.idx = 0
        self.render()

    def go_last(self):
        self.idx = len(self.entries) - 1
        self.render()


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("folder", nargs="?",
                   default=r"C:\_backup\github\khub-knowledge-fabric\usecases\arc-agi-3\.tmp\playlogs\20260409-123632_ep01")
    p.add_argument("--level", type=int, default=None,
                   help="Filter to only this levels_completed value (0=L1, 1=L2, 2=L3)")
    args = p.parse_args()

    print(f"Loading playlog from: {args.folder}")
    entries = load_playlog(args.folder, level_filter=args.level)
    if not entries:
        print("No frames found (try --level 0 / 1 / 2)")
        sys.exit(1)
    print(f"Loaded {len(entries)} frames. Level filter: {args.level}")
    print("Controls: Right/Space=next  Left=prev  Home=first  End=last  Q=quit")
    PlaylogViewer(entries)


if __name__ == "__main__":
    main()
