# Lossless compression specialists v0

Status: implemented local research baselines, consultative only. No production,
Godot, Zig or Fovea runtime integration. No claim of superiority on real assets.

## Scope and methods

- `bellium/hybrid/image-compression:v0`: L, RGB or RGBA byte pixels with channel
  planes. Width, height, mode, invisible RGB values and alpha survive exactly.
  File metadata, ICC profiles and EXIF are outside the pixel API.
- `bellium/hybrid/stream-compression:v0`: independent byte chunks. All predictor
  state resets per packet; the caller owns framing, ordering and transmission.
- `bellium/hybrid/splats-compression:v0`: static finite little-endian float32
  records and optional temporal XOR residuals. The base fingerprint and record
  count must match. The returned record bytes, including signed zero, are exact.
- `bellium/knn/byte-compression:v0`: causal three-neighbor predictor with a
  64-example memory of two-byte contexts, deterministic recency tie-breaking.
- `bellium/micro-nn/byte-compression:v0`: one adaptive linear neuron, three inputs
  plus bias (four Q12 weights). Integer online updates learn only from bytes
  already reconstructed. There are no pretrained weights, external model files,
  hidden layers or claims of an image/splat autoencoder.

Both learned predictors produce modulo-256 byte residuals, then zlib encodes the
residual stream. They reset every 4096 bytes. Decoding repeats the same causal
updates from the specified initial state; no source bytes are leaked to a model.
Algorithms and reset intervals are part of the versioned wire contract.

`method="auto"` actually encodes eligible candidates (raw, zlib, byte delta +
zlib, k-NN + zlib, micro-NN + zlib), then chooses the smallest full packet.
Ties prefer this listed order. This optimizes bytes, not latency. Forced methods
may expand data. Even the automatic packet can exceed the original bytes because
metadata and checksums cost space. `inspect_packet` reports that overhead.

The cap is 16 MiB decoded per packet. Learned methods are limited to 64 KiB
per packet for CPU cost; automatic mode skips them above this limit. Chunking
larger inputs is caller-owned. The current implementation uses Python CPU only.

## Research format BLCP/1

This is an original Bellium research envelope, **not a .fovea file or a replacement
for its codec/protocol**. The 56-byte splat record is 14 float32 fields: position
xyz, rotation quaternion xyzw, scale xyz, opacity, RGB. It excludes SH coefficients,
IDs, bones and additional application fields. Callers must preserve record order
and supply the exact base for temporal decoding. Temporal residuals are not an
animation rig, motion estimation model or deformation model.

Header (`<4sBBIII32s`): `BLCP`, version 1, method index, decoded byte count,
metadata byte count, payload byte count, SHA-256 of preprocessed decoded bytes.
Canonical UTF-8 JSON metadata follows, then payload, then SHA-256 of everything
before that final digest. Method indices are raw=0, zlib=1, delta=2, knn=3,
micro-nn=4. These are corruption checks, not authentication.

Decoders validate lengths and method before decompression, cap inflation, reject
truncated/trailing compressed data, and verify decoded bytes. Images validate
dimensions/mode. Splats validate finite fields, layout, count, reconstructed
record digest and optional base digest. No pickle, implicit file writes or network.

## Usage

```python
from bellium.compression import encode_image, decode_image, encode_stream, decode_stream
from bellium.specialists.compression import compress

packet = encode_stream(b"sensor=12;" * 100, method="micro-nn")
assert decode_stream(packet) == b"sensor=12;" * 100
image_packet = encode_image(b"\xff\x00\x00\x00", width=1, height=1, mode="RGBA")
assert decode_image(image_packet).pixels == b"\xff\x00\x00\x00"
result = compress(b"sensor=12;" * 100, kind="stream")
# Consultative result with packet bytes and measured sizes; confidence is None.
```

For splats, `encode_splats(records, base=previous_records)` and
`decode_splats(packet, base=previous_records)` operate on explicitly supplied
56-byte records. Python floats are not silently converted by this API.

## Evidence and next gate

`tests/test_compression.py` covers fresh-state reconstruction, images including
transparent colored pixels, float32 splats, a temporal sequence, invalid bases,
corrupt/oversized packets and bounded decompression. Synthetic measurements show
behavior on those fixtures; they do not demonstrate a production compression win.

Run `python scripts/benchmark_compression.py --output <report.json>` for all
methods, packet sizes, encode/decode timing medians, plain zlib and PNG references
when Pillow is installed. Motion comparisons include the base packet cost.

Next gate: independent licensed image/splat assets and representative sanitized
data streams; compare total bytes, encode/decode cost and memory against PNG,
applicable stream codecs and the existing .fovea codec. A fixed-weight pretrained
nonlinear model, SH-aware codec or consumer integration requires its own evidence.

### First measured outcome (15 September 2026)

On the ten synthetic fixtures (three streams, three RGBA images, two static
splat buffers and two temporal splat buffers), all six requested methods
reconstruct exactly. Automatic selection picks raw twice, zlib four times and
delta four times. Neither learned predictor wins against the best classical
candidate on this corpus. They also incur substantially higher Python CPU cost.
PNG is smaller than BLCP on all three image fixtures. This is a negative result
for a learned-codec advantage, not a reason to claim the codecs are incomplete.
Production assets, learned nonlinear predictors and .fovea comparisons remain
separate, unvalidated work.

### Independent corpus (16 September 2026)

The [next evaluation](../COMPRESSION_INDEPENDENT_V1.md) covers 20 source-disjoint
samples from documented photographs, public tabular data and a separately
trained CC0-derived synthetic-scene splat asset. Exact reconstruction holds for
all supported sample bytes. The existing micro-NN wins once on a development
crop, but never on the held-out set. One spatial-neuron experiment fails its
development gate and remains outside automatic selection and the catalog.
