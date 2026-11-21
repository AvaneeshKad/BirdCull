# BirdCull

A local app that helps a bird photographer cull thousands of RAW photos. It finds the shots that are blurry, have no bird in them, or are near-duplicates, and suggests which to delete. You review everything before anything is removed.

Everything runs on your own computer. Photos are never uploaded anywhere.

## How it works

1. **Scan.** Pick the folders to check. BirdCull reads each RAW file, finds birds with a YOLO detector (GPU if available, otherwise CPU), and measures how sharp the bird is.
2. **Review.** It suggests photos to remove. Every suggestion can be kept with one click, and photos can be rotated. Bursts of near-identical shots keep only the best few frames.
3. **Finish.** Chosen photos go to the Recycle Bin by default. Permanent delete is also available, but you must type `DELETE` to confirm.

Safety:

- Nothing is moved or deleted until the final step.
- Photos you keep are remembered and never suggested again.
- Every removal is appended to `removed_photos_log.csv`.
- It refuses to touch paths outside the folders you chose.
- The scan can be stopped and resumed.
- Windows is kept from sleeping during a scan.

## Install

### Windows 11 (NVIDIA GPU recommended)

1. Copy the `app/` folder to the PC.
2. Double-click `install.bat`. It needs internet and takes 10-20 minutes. It installs Python, PyTorch (CUDA 12.8, which RTX 50-series cards need) and the other dependencies, then downloads the detector model (about 51 MB). It also puts a **BirdCull** icon on the Desktop.
3. Open BirdCull from the Desktop icon. Your browser opens at `http://localhost:8787`.

`BirdCull_Guide.pdf` is a short picture guide for non-technical users.

> The Windows installer has not been tested on a real Windows machine yet. Try it first on a small folder of photos, and keep a backup of your favourites.

### Linux / macOS

```bash
cd app
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision          # pick the build for your GPU from pytorch.org
pip install -r requirements.txt
python run_birdcull.py
```

The model downloads automatically on first run.

## Settings

| Environment variable | Meaning | Default |
|---|---|---|
| `PORT` | Port for the local server | `8787` |
| `BIRDCULL_DATA` | Where settings and scan results are stored | `~/BirdCull` |
| `NO_BROWSER` | Set to skip opening the browser | unset |

## Repository layout

```
app/                  the BirdCull app
  birdcull/
    core.py           scanning, sharpness scoring, keep/remove decisions
    server.py         local web server and API
    index.html        the 3-step user interface
  install.bat         Windows installer
  run.bat             Windows launcher
  requirements.txt
review2/              review page for the blurry/ and duplicates/ folders
*.py (top level)      scripts from the original overnight cull of one photo library
BirdCull_Guide.pdf    picture guide
```

The top-level scripts are kept for history. They contain paths for the author's own drive and are not needed to run the app.

## Notes

- The detector is [Ultralytics](https://github.com/ultralytics/ultralytics) YOLO (`yolo26l.pt`). Check the Ultralytics licence (AGPL-3.0 unless you have an enterprise licence) before redistributing the app or its weights.
- Sharpness is measured on a real RAW decode, not the embedded JPEG preview, so it reflects actual focus.
- There is no cloud or multi-user mode. It is a single-user desktop tool.
