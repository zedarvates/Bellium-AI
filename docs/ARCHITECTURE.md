# Architecture and compatibility

The preferred APIs return `SpecialistResult`: proposal, confidence, abstention,
authority mode and evidence. They live in `bellium.knn`, `bellium.micro_nn` and
`bellium.specialists`. Registry membership does not authorize an action.

The published APIs from Bellium commit
`9ab60d91a9b78829526c3a99f35d87fcadf56bc2` are retained:

- `legacy.micro_nn`: historical features, CLI and calibration; inference delegates
  to the validated Bellium MLP core. Legacy weight bytes remain pinned.
- `legacy.knn_asset_quality`: original report and storage schema, still shadow-only.
- `bellium.cutout`: Pillow RGB/RGBA cutout and normalization, via the `image` extra.
- `bellium.inpaint`: Pillow adapter over the corrected patch implementation.
- `bellium.language.PhonemeMemory`: explicit-record compatibility API; inference
  needs a language filter when the memory contains several languages.
- `bellium.routing.HybridRouter`: declarative capability proposals, no execution.

The compatibility functions retain their own result types. They do not become
registry-authorized specialists merely by being imported. In particular,
`normalize_background` raises `UnsafeCutoutError` carrying metrics if extraction
requires review or escalation.

The canonical editable model tree is `models/`. Building a wheel copies its JSON
files into `bellium/_data/models/`; `bellium.resources.model_path` resolves these
resources without depending on the current directory. New specialists must use
that helper, not `Path(__file__).parents[2] / "models"`.

Asset memories are bounded local JSONL ledgers. They assume one writer, store
verified reports rather than raw assets, and reject writes when the size budget
is exhausted. Automatic rotation and multi-process coordination are not supplied.
