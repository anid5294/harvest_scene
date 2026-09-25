# Harvest MuJoCo

Fixed-base Unitree G1 and Dex3 harvesting tests built on
[OrchardBench](https://github.com/humphreymunn/orchardbench). The current gate
commands the 43-joint robot to detach one apple from a generated tree and place
it in a tray.

## First harvest test

The test passes only when all of these conditions hold:

1. the full G1/Dex3 model loads with the contract's 43 joints;
2. the right arm reaches a selected apple;
3. the apple detaches through OrchardBench's force-based stem model;
4. the robot transports and releases the apple;
5. the apple settles inside the physical tray; and
6. a valid 30 Hz state/action trace and headless MP4 are written.

This is a feedback-gated scripted policy using simulator state. It is not a
learned or vision policy. The robot base is fixed for this test.

## Workstation setup

Use sibling checkouts so this repository does not modify OrchardBench:

```text
work/
├── harvest_scene/
├── orchardbench/
└── .tools/bin/pixi
```

Clone and pin the dependencies:

```sh
git clone https://github.com/anid5294/harvest_scene.git
git -C harvest_scene switch --track origin/feature/orchardbench-g1-contract

git clone https://github.com/humphreymunn/orchardbench.git
git -C orchardbench checkout 6313313db8b1a7d23fb2cc3afd67cac46f29399a
```

Install Pixi under the work directory if it is not already available:

```sh
mkdir -p .tools/bin
curl -fsSL \
  https://github.com/prefix-dev/pixi/releases/download/v0.81.0/pixi-x86_64-unknown-linux-musl.tar.gz \
  | tar -xz -C .tools/bin
export PATH="$PWD/.tools/bin:$PATH"
```

The workstation also needs an NVIDIA GPU, `xvfb-run`, and `ffmpeg`. Confirm the
checkout and warm the OrchardBench environment before running the gate:

```sh
cd harvest_scene
python3 orchard.py check
cd ../orchardbench
pixi run python -c 'import newton, warp; print("OrchardBench environment ready")'
cd ../harvest_scene
```

## Run

```sh
python3 orchard.py g1-harvest
```

The command exits with status `0` only on success. It writes:

- `artifacts/g1_harvest_seed42.mp4` — 960 × 540 headless overview;
- `artifacts/g1_harvest_seed42.json` — result, transitions, and provenance;
- `artifacts/g1_harvest_seed42.npz` — contract-ordered state/action trace; and
- `artifacts/runs/*_g1-harvest.json` — launcher manifest.

The trace is intentionally marked `training_eligible: false` until the three
required robot camera streams are recorded. It does not fabricate camera data.

## Other checks

```sh
python3 orchard.py tree --frames 120   # OrchardBench physics
python3 orchard.py g1                  # G1/tree assembly render
python3 contract.py --self-test        # data-contract validation
```

The standalone tabletop demo remains available with a local Python environment:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python demo.py --headless
```

## Repository layout

```text
g1_harvest.py          G1 apple-detach and tray-placement gate
g1_orchard.py          G1/orchard assembly render
orchard.py             Pinned launcher and run provenance
contract.py            Canonical joint and camera data validation
demo.py                Standalone YCB tabletop pick-and-place demo
assets/                 Vendored G1 and YCB apple assets
tests/                  Deterministic geometry and contract checks
```
