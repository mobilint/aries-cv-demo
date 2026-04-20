# ARIES CV Demo

ARIES CV Demo is a tiled multi-channel computer vision demo for ARIES systems. It loads demo manifests from `assets/*/config/demo.yaml` and runs Mobilint QB Runtime models on MLA accelerators.

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
- `video/`: sample videos when bundled

The runtime selects `mla400` when at least 4 accelerators are available, otherwise `mla100` when at least 1 accelerator is available.

Current parser fields:

- Model settings: `pipeline_type`, `input_type`, `mxq_path`, `device`, `num_core`, `core_id`, `pipeline_config`
- Feeder settings: `type`, `sources`
- Layout settings: `canvas_size`, `preview_asset`, `splash_assets`, `background_images`, `worker_tiles`

## Controls

- Launcher: `Up` / `Down` to select, `Enter` to run, `q` or `Esc` to quit
- Runtime: `D` toggle FPS, `T` toggle elapsed time, `M` fullscreen, `C` stop workers, `F` start workers, `Q` or `Esc` quit
- Mouse: left click enables a worker, right click disables a worker

## update.sh

`./update.sh` is a setup-and-build helper. It installs system dependencies, adds Mobilint's APT repository, installs Mobilint packages, runs `git pull` and optional `git lfs pull`, builds the project, and updates the desktop shortcut and icon.

## Packaging

```bash
./package/package.sh aries2-v4 aries2
```

See [`package/README.md`](/home/beomsun/projects/aries-cv-demo/package/README.md) for package-local build details.
