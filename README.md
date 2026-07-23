# ARIES CV Demo

ARIES CV Demo is a tiled multi-channel computer vision demo for ARIES systems. It loads demo manifests from `assets/*/config/demo.yaml` and runs Mobilint QB Runtime models on MLA accelerators.

The runtime follows a producer/worker/display pipeline: feeder threads continuously publish the latest frames into per-feeder buffers, bounded model/core worker loops consume new frames for inference, and rendered worker results are delivered through a display queue. This keeps capture I/O decoupled from model execution and lets `fire_detection` and `weapon_detection` run as single-model multi-channel demos without creating one inference thread per tile.

Built-in demos:

- `fire_detection`
- `weapon_detection`

## Requirements

- Ubuntu 20.04+ recommended
- CMake 3.16+
- `build-essential`
- `libopencv-dev`
- OpenMP toolchain support
- Mobilint QB Runtime and ARIES driver

The root build uses `OpenCV`, `OpenMP`, `yaml-cpp`, and `qbruntime`. On Linux, `qbruntime` must be installed on the system or provided with `-DQBRUNTIME_PATH=<path>`.

## Build

```bash
mkdir -p build
cd build
cmake ..
cmake --build . -j"$(nproc)"
```

Executable:

```bash
./build/src/demo/demo
```

## Run

Start the launcher:

```bash
./build/src/demo/demo
```

List demos:

```bash
./build/src/demo/demo --list
```

Run a demo directly:

```bash
./build/src/demo/demo fire_detection
./build/src/demo/demo weapon_detection
```

You can also use:

```bash
./run.sh
```

## Configuration

Each demo is stored under `assets/<demo-name>/`:

- `config/demo.yaml`: manifest
- `config/LayoutSetting_*.yaml`: tiled layout
- `config/FeederSetting_*.yaml`: input sources
- `config/ModelSetting_*.yaml`: model and device assignment
- `layout/`: background images
- `mxq/`: model files
- `video/`: sample videos downloaded from Hugging Face Buckets

The runtime selects `mla400` when at least 4 accelerators are available, otherwise `mla100` when at least 1 accelerator is available.

### Model post-processing

- `weapon_detection` uses a YOLO26s-based weapon model with two class labels. Its model settings must use `input: uint8`, `post: dflfree`, and `num_classes: 2` so it routes through the DFL-free object postprocessor.
- `fire_detection` uses a YOLO11s-based fire model with two class labels. Its model settings must use `input: float32`, `post: anchorless`, `decode_bbox: true`, and `num_classes: 2` so it routes through the anchorless YOLO11 postprocessor.

When changing these configs, keep the demo shape as multi-channel with one model per demo mode; `worker_tiles` can all reference `model_index: 0` on MLA100, while MLA400 modes may map tiles across the four configured model/device entries.

### Runtime scheduling

The default worker scheduler is model/core bounded. Each logical worker tile keeps its own feeder index, model index, ROI, score smoothing state, and latest frame index, but inference execution is handled by `num_core` worker loops per configured model instead of one OS thread per tile. For example, `fire_detection` MLA400 has 96 feeder entries and 96 worker tiles mapped to four model entries with eight cores each, so the demo creates 96 feeder producer threads and 32 inference worker threads by default. This preserves the multi-channel producer/display-queue structure while avoiding the 96 inference threads that a tile-per-thread policy would create.

## Download Large Assets

Large `.mxq` model files and `.mp4` sample videos are not stored in Git or Git LFS. Download them from Hugging Face Buckets before building or running demos:

```bash
./download.sh
```

On Windows:

```bat
download.bat
```

Default buckets:

- Weapon detection: `hf://buckets/mobilint/aries-weapon-detection-demo-assets`
- Fire detection: `hf://buckets/mobilint/aries-fire-detection-demo-assets`
- Ultralytics demo: `hf://buckets/mobilint/aries-ultralytics-demo-assets`

The downloader preserves the runtime directory layout used by the YAML configs:

- `assets/weapon/mxq/`, `assets/weapon/video/`
- `assets/fire/mxq/`, `assets/fire/video/`
- `assets/ultralytics/mxq/`, `assets/ultralytics/video/`

Optional environment variable overrides:

- `HF_WEAPON_ASSET_BUCKET_ID` for weapon assets
- `HF_FIRE_ASSET_BUCKET_ID` for fire assets
- `HF_ULTRALYTICS_ASSET_BUCKET_ID` for ultralytics assets
- `HF_DOWNLOAD_VENV_DIR` for the local Hugging Face CLI virtualenv directory

The scripts use `uv` to create a local `.hf_venv` and install `huggingface-hub`. Hugging Face credentials, if required for private buckets, should be provided via `hf auth login` or environment configuration and must not be committed.

Files directly under `mxq/` and `.mp4` files under `video/` are synchronized by the asset scripts. If non-`.mp4` files are placed under an asset `video/` directory, `upload.sh` and `upload.bat` will not upload them unless the include patterns are updated.

### Hugging Face Bucket Operations

The production buckets must be owned by the Hugging Face `mobilint` organization:

- `mobilint/aries-weapon-detection-demo-assets`
- `mobilint/aries-fire-detection-demo-assets`
- `mobilint/aries-ultralytics-demo-assets`

Migration checklist for maintainers with write access:

1. Rename or migrate the existing vision VLM demo bucket content from `mobilint/aries-vision-vlm-demo-assets` to `mobilint/aries-weapon-detection-demo-assets`.
2. Create `mobilint/aries-fire-detection-demo-assets` under the `mobilint` organization.
3. Create `mobilint/aries-ultralytics-demo-assets` under the `mobilint` organization when ultralytics assets are maintained for this repository.
4. Upload assets using the same layout expected by this repository:
   - Weapon: `assets/weapon/mxq/*` -> `mxq/*`, `assets/weapon/video/**/*` -> `video/**/*`
   - Fire: `assets/fire/mxq/*` -> `mxq/*`, `assets/fire/video/**/*` -> `video/**/*`
   - Ultralytics: `assets/ultralytics/mxq/*` -> `mxq/*`, `assets/ultralytics/video/**/*` -> `video/**/*`
5. Use `./upload.sh` or `upload.bat` after authenticating with the Hugging Face CLI to synchronize local assets to the corresponding buckets. These upload scripts run `hf buckets sync --delete`, so remote files matching the included `mxq/*` and `video/**/*.mp4` asset patterns are deleted when they are missing locally. `.gitkeep` placeholders are excluded from uploads.
6. Verify downloads with `./download.sh` after authenticating with the Hugging Face CLI.

Do not store Hugging Face tokens or credentials in this repository.

Current parser fields:

- Model settings: `pipeline_type`, `input_type`, `mxq_path`, `device`, `num_core`, `core_id`, `pipeline_config` including `draw_label_text` / `draw_score_text`
- Feeder settings: `type`, `sources`
- Layout settings: `canvas_size`, `preview_asset`, `splash_assets`, `background_images`, `worker_tiles`

## Controls

- Launcher: `Up` / `Down` to select, `Enter` to run, `q` or `Esc` to quit
- Runtime starts fullscreen with elapsed time hidden by default.
- Runtime: `D` toggles FPS overlay (`ultralytics`: per-tile FPS, `weapon_detection`/`fire_detection`: total average FPS), `T` toggles elapsed time, `M` fullscreen, `C` stop workers, `F` start workers, `Q` or `Esc` quit
- Mouse: left click enables a worker, right click disables a worker and restores the tile from the layout background

## Performance validation

After changing runtime scheduling or post-processing, validate both production demos with the same hardware/assets before and after the change when possible:

```bash
./build/src/demo/demo --list
./build/src/demo/demo --debug weapon_detection
./build/src/demo/demo --debug fire_detection
```

Check that all tiles update, `C`/`F` and mouse enable-disable still work, disabled tiles are restored from the layout background, and quit exits cleanly. Compare average display FPS with debug NPU timings separately. The NPU metric must measure only `qbruntime::Model::infer()`; preprocessing, post-processing, rendering, queue handoff, and display composition are CPU-side costs and should be read from the separate debug `pre/post/draw` timings or inferred from end-to-end FPS. The queue-based architecture adds synchronization and one worker-output handoff, so possible regression sources are queue growth, extra `cv::Mat` copies, and workers spinning on unchanged frames. The implementation mitigates these by consuming only latest feeder frames, sleeping briefly when no new frame exists, reusing per-worker model workspaces, and draining all pending worker results each display refresh.

## update.sh

`./update.sh` is a setup-and-build helper. It installs system dependencies, adds Mobilint's APT repository, installs Mobilint packages, installs/checks `uv`, runs `git pull`, downloads CV assets with `./download.sh`, builds the project, and updates the desktop shortcut and icon.

## Packaging

```bash
./package/package.sh aries2-v4 aries2
```

See [`package/README.md`](/home/beomsun/projects/aries-cv-demo/package/README.md) for package-local build details.
