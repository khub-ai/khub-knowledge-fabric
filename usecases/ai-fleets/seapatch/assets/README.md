# Assets

This directory holds example images for documentation and the Getting Started section of README.md.

## Required files

| File | Source | Description |
|---|---|---|
| `scout_failure_frame.jpg` | SeaDronesSee val split | A person-in-water frame that a baseline classifier scores as "whitecap" with ≥0.70 confidence. Select using `pool_builder.py --select-hardest --failure-dir assets/`. |
| `thermal_confirmation.png` | FLIR Free Thermal Dataset or oracle | Commander drone FLIR return showing the corresponding 37°C heat signature at the same coordinates. For simulation, a screenshot of the thermal_oracle output description is sufficient. |

## How to populate

After downloading SeaDronesSee:

```bash
python python/pool_builder.py \
    --dataset-root data/seadronessee/ \
    --split val \
    --pool-dir data/pool/ \
    --select-hardest \
    --failure-dir assets/
```

This copies the hardest-to-detect person frame to `assets/failure_0000.jpg`.
Rename it to `scout_failure_frame.jpg` for use in documentation.
