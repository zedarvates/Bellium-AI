# Bellium AI usage examples

These examples run locally against the APIs in this checkout. They need no API
key, model download, server or agent framework. Commands below run from the
repository root with Python 3.10+.

```sh
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r examples/requirements.txt
```

| Example | What it demonstrates |
| --- | --- |
| [Image tool](tools/image_tool.py) | A CLI and callable adapter: grayscale, sepia, Bellium cutout and small masked-fill previews. |
| [Tool routing](tool_routing.py) | Register capabilities, request an offline tool and handle unsupported requests without executing a proposal. |
| [Micro-NN triage](micro_nn_triage.py) | Extract named features, consult an imported classifier and apply a consumer-side abstention threshold. |
| [Agent skill](skills/bellium-local-tools/SKILL.md) | A reusable skill describing the actual tools, commands, limits and result handling. |

## Try it without supplying an image

```sh
python -m examples.tools.image_tool demo --output-dir output/bellium-demo
python -m examples.tool_routing
python -m examples.micro_nn_triage
```

The image demo creates an original synthetic graphic, a damaged copy, a mask,
and `grayscale.png`, `sepia.png`, `cutout.png`, `inpaint.png`. It prints a JSON
report. Use a new directory to rerun; existing files are never overwritten.
These fixtures demonstrate wiring, not photographic quality.

The routing demo proposes `image.sepia` for the matching request and returns
`null` for unsupported requests. The micro-NN example uses a synthetic error;
its label/score come from the included legacy weights. The score is not a
calibrated probability of correctness. Neither example performs an action.

With the current weights, the missing-file example suggests `network` with a
score around `0.57`; the `0.8` threshold therefore abstains. This deliberately
exposes a model error instead of presenting successful execution as accuracy.

## Use the image tool with your own files

```sh
python -m examples.tools.image_tool grayscale --input input.png --output output/gray.png
python -m examples.tools.image_tool sepia --input input.png --output output/sepia.png --strength 0.7
python -m examples.tools.image_tool sepia --input input.png --mask selection.png --output output/local-sepia.png
python -m examples.tools.image_tool cutout --input input.png --output output/cutout.png
python -m examples.tools.image_tool inpaint --input small.png --mask mask.png --output output/fill.png
```

- Grayscale and sepia are deterministic Pillow filters in the example adapter,
  not trained Bellium models. Black and white here means grayscale, not binary
  thresholding. Sepia is an adjustable warm duotone.
- Filter masks use `0` to preserve, `255` to apply, and intermediate values to
  blend. Images and masks must have the same dimensions. Alpha is preserved.
- Cutout estimates a foreground mask and saves a candidate only when the
  published heuristic recommends `confident`; inspect the result yourself.
- Inpaint accepts a binary selection (`mask > 128`), enforces routing **before**
  filling and preserves pixels outside the mask. It caps the input at 16,384
  pixels and the mask at 5%. It is a preview for small defects, not a complete
  magic eraser; mask size alone does not establish reconstruction quality.
  Bounded bilinear interpolation first checks every known pixel in an enclosing
  rectangle; otherwise context matching uses intact original patches and three
  donors. Reports expose the chosen method and interpolation gate. Core
  abstention also suppresses output; see [the comparison and remaining hidden-detail losses](../docs/INPAINT_PREVIEW_QA.md).
- All inputs are limited to 1,048,576 pixels, output is PNG, and existing files
  are refused. These are bounded examples, not high-resolution editor APIs.
- Exit codes: `0` completed/candidate written, `1` invalid input or I/O error,
  `2` abstained with no output image. `review_required` distinguishes candidates.

## Call it from an editor or agent tool

Import the adapter from an application running in this checkout:

```python
from examples.tools.image_tool import edit_image

report = edit_image("sepia", "input.png", "output/sepia.png", strength=0.7)
if report["status"] == "abstained":
    print(report["reason"])
elif report["review_required"]:
    print("Candidate to preview:", report["output"])
else:
    print("Result:", report["output"])
```

An agent framework can expose this function as a tool with the four-operation
allowlist. Supply user-selected paths through the host application and display
the returned status; do not execute arbitrary command text from a model.
This adapter has no filesystem sandbox. A hosted integration must enforce its
own allowed input/output directories before calling it.

## Use the sample skill

An agent can read [SKILL.md](skills/bellium-local-tools/SKILL.md) directly. To make
it discoverable by Codex, copy the `bellium-local-tools` folder into your chosen
Codex skills directory (normally `~/.codex/skills/`). Keep this repository
available and provide its location; copying the skill does not install Bellium.
No skill or application configuration is changed by these examples.

Example requests:

- “Use Bellium to make a sepia copy of this image at 70% strength.”
- “Use Bellium to preview a fill for this small mask and show the report.”
- “Show how Bellium proposes an offline tool and abstains on an unknown task.”

## Verify the examples

```sh
python -m unittest discover -s examples/tests -v
python run_all_tests.py
```

The example tests exercise CLI execution, source/alpha/mask preservation,
overwrite refusal, abstention and consultative routing/triage. They do not
certify real-image quality, calibrated model scores or real-time performance.
Physics, material-map extraction and a complete image editor remain planned in
[the roadmap](../docs/ROADMAP.md). Dedicated experimental nano/micro alpha
classifiers are available separately in [the alpha-QA laboratory](../docs/ASSET_FACTORY_QA.md).
