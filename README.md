# Harvest MuJoCo

A MuJoCo tabletop pick-and-place demo using a Unitree G1 humanoid with two
7-DoF dexterous hands and a YCB apple. A rule-based feedback policy grasps the
apple, lifts it, transfers it, and places it in a tray.

The G1 pelvis is currently fixed. The policy uses MuJoCo state directly; 
it is not learned and does not use camera perception.

## Setup

Python 3.11 or newer recommended. 

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

Interactive viewer:

```sh
.venv/bin/mjpython demo.py
```

On non-macOS platforms, `.venv/bin/python demo.py` can be used instead.

Headless verification:

```sh
.venv/bin/python demo.py --headless
```

The process exits with status `0` only when the apple is released, settled
inside the tray, and MuJoCo reports no warnings. An optional JSON report can be
written with `--report PATH`.

## OrchardBench integration

`orchard.py` checks its exact revision before every run and stores run manifests under `artifacts/`.

```sh
python orchard.py check
python orchard.py tree --frames 120
python orchard.py view
python orchard.py usd
python orchard.py harvest
python orchard.py g1
```

`python orchard.py g1` is the first G1/OrchardBench integration gate. It imports
the vendored full G1 + dual Dex3 model with a fixed base, verifies the 43
contract joints, places it beside a seed-42 fruit tree, and writes a short
headless GIF plus JSON report under `artifacts/`.

`contract.py` is the executable `g1_29body_dex3_43d_v1` interface. Robot,
camera, recorder, and hardware adapters must pass its validation before their
episodes can be used for training.

```sh
python contract.py --self-test
python contract.py
```

## Repository layout

```text
demo.py                 Environment, policy, rollout, and verification
contract.py             Canonical 43-channel and camera schema validation
orchard.py              Pinned OrchardBench launcher and run provenance
g1_orchard.py            Fixed-base G1 + orchard assembly and headless render
assets/unitree_g1/      Vendored G1 MJCF and referenced meshes
assets/ycb/013_apple/   YCB apple mesh and texture
requirements.txt        Python dependencies
```
