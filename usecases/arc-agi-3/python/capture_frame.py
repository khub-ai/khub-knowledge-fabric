"""
capture_frame.py — capture the initial frame of any ARC-AGI-3 environment as a PNG.

Usage:
    python capture_frame.py --env tr87 --out tr87_level1.png
    python capture_frame.py --env ls20 --out ls20_level1.png
"""
import sys, argparse, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parents[2]))

_key_file = Path("P:/_access/Security/api_keys.env")
if _key_file.exists():
    for _line in _key_file.read_text().splitlines():
        for _prefix in ("ANTHROPIC_API_KEY=", "arc_api_key=", "TOGETHER_API_KEY="):
            if _line.startswith(_prefix):
                _var = _prefix.rstrip("=").upper()
                if not os.environ.get(_var):
                    os.environ[_var] = _line.split("=", 1)[1].strip()

import arc_agi
from agents import obs_frame
from render_replay import frame_to_image, add_label


def save_frame(env_id: str, out: Path, scale: int = 8) -> None:
    arc = arc_agi.Arcade(arc_api_key=os.environ.get("ARC_API_KEY", ""))
    env = arc.make(env_id)
    obs = env.reset()
    frame = obs_frame(obs)
    img = frame_to_image(frame, scale=scale)
    img = add_label(img, f"{env_id} — level 1 (initial frame, {len(frame[0])}x{len(frame)} grid)")
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"Saved: {out}  ({img.width}x{img.height}px)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env",   default="tr87")
    p.add_argument("--out",   default=None)
    p.add_argument("--scale", type=int, default=8)
    args = p.parse_args()
    _tmp = Path(__file__).parent.parent / ".tmp"
    out = Path(args.out) if args.out else _tmp / f"{args.env}_level1.png"
    save_frame(args.env, out, scale=args.scale)


if __name__ == "__main__":
    main()
