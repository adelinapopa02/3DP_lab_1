# 3D Data Processing – Lab 1: SGM Stereo Matching with Monocular Depth Initial Guess

This project implements **Semi-Global Matching (SGM)** with a **Census transform** cost, initialized/refined using a **monocular depth estimate**, to compute dense disparity maps from rectified stereo image pairs.

The pipeline:

1. Load a rectified stereo pair (`left`/`right`) plus the corresponding monocular depth cues (`left_mono`/`right_mono`).
2. Compute the matching cost volume via the **Census transform**.
3. Aggregate costs along multiple directions (**Semi-Global Matching**).
4. Produce a dense **disparity map**, comparable against the provided ground truth (`rightGT`).

---

## Author

* [@adelinapopa02](https://github.com/adelinapopa02)

---

## Project overview

| File | Role |
|---|---|
| `sgm.h` / `sgm.cpp` | Core SGM-Census implementation: cost computation, cost aggregation, disparity selection. |
| `main.cpp` | CLI entry point: loads a data folder, runs SGM, writes the output disparity image. |
| `Examples/` | Test stereo pairs (`Aloe`, `Cones`, `Plastic`, `Rocks1`) with left/right images, monocular cues and ground-truth disparity. |
| `Stereo Matching SGM + Monocular.pdf` | Assignment write-up / report. |

---

## Build (out-of-tree)

```bash
mkdir build
cd build
cmake ..
make
```

## Usage (from `build/`)

```bash
./sgm <data folder> <output disparity file> <disparity range>
```

### Example

```bash
./sgm ../Examples/Rocks1/ out.png 85
```

Runs SGM-Census on the `Rocks1` stereo pair with a disparity search range of 85 pixels and writes the resulting disparity map to `out.png`.
