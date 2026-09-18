---
name: bellium-local-tools
description: Use a Bellium-AI checkout for local grayscale or sepia filters, simple cutout previews, small masked fill previews, or demonstrations of tool routing and micro-NN error triage.
---

# Bellium local tools

Run commands from the Bellium-AI repository root using Python 3.10+ and Pillow.
Locate the user's checkout; the skill does not bundle the library. If needed,
install the example dependencies with `python -m pip install -r examples/requirements.txt`
in the environment chosen for this repository.

For a self-contained demonstration with generated synthetic inputs:

```sh
python -m examples.tools.image_tool demo --output-dir output/bellium-demo
```

Use a new output directory on subsequent runs. For a requested image operation:

```sh
python -m examples.tools.image_tool grayscale --input input.png --output output/gray.png
python -m examples.tools.image_tool sepia --input input.png --output output/sepia.png --strength 0.7
python -m examples.tools.image_tool binary --input input.png --output output/binary.png --threshold 128
python -m examples.tools.image_tool cutout --input input.png --output output/cutout.png
python -m examples.tools.image_tool inpaint --input small.png --mask mask.png --output output/fill.png
```

Choose the operation the user requested. Grayscale, sepia and binary threshold
use the deterministic `bellium.filters` core, support an optional same-size
grayscale mask and preserve alpha; binary maps values at or above the threshold
to white. The unified CLI exposes the same filters, for example
`python -m bellium.cli filter sepia input.png -o output/sepia.png --strength 0.7`.
Cutout works best with a plain background. Inpaint is an experimental
small-fill preview: at most 16,384 image pixels and 5% selected pixels, with
mask values above 128 selecting the fill. This is not a general magic eraser.

Read the JSON report and process exit code: `0` means completed or candidate
written, `1` is an error, and `2` means abstention with no output image.
Show generated candidates for review and retain the original. The tool refuses
existing output files. An abstention does not authorize calling a remote model.
Do not relabel heuristic scores as calibrated confidence or measured quality.

For integration demonstrations:

```sh
python -m examples.tool_routing
python -m examples.micro_nn_triage
```

The router proposes a tool without executing it. Micro-NN triage returns a
consultative label and may abstain; it never repairs a system. Physics, PBR map
extraction and nano-NN specialists remain roadmap items, not callable tools.
