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

## Repository layout

```text
demo.py                 Environment, policy, rollout, and verification
assets/unitree_g1/      Vendored G1 MJCF and referenced meshes
assets/ycb/013_apple/   YCB apple mesh and texture
requirements.txt        Python dependencies
```

