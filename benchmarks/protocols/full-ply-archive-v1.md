# Complete PLY archival gate — 17 September 2026

Objective: reconstruct the full supported PLY file byte-for-byte, including
header, field order, SH coefficients and unknown scalar attributes. No field
interpretation, normalization, quantization or truncation. Preserve BLCP/1 code.

Bounded supported layout: binary little- or big-endian PLY 1.0, one vertex
element, scalar properties, <=256 properties, <=16 KiB header, <=16 MiB file.
ASCII, list properties, faces/additional elements and malformed counts fail.

Compare original interleaved records and byte-plane transposition, with raw,
zlib, delta+zlib, and automatic selection. Existing k-NN/micro-NN retain their
64 KiB cap and are tested on small full-file fixtures; do not extrapolate a
neural advantage from classical large-file results.

Use the previously reviewed complete Horse Statue derived reconstruction:
SHA256 `4392920074f41204ec900244da06505a8696312ea989340df88b1bd9c025806a`.
It has a CC0 source but comes from synthetic multiview footage. This is a known
integration asset, not a new independent holdout or a natural motion sequence.

Measure full packet bytes, encode/decode medians over three runs, exact file
digest, a bare zlib reference and an original-layout BLCP zlib reference. Measure
Python-traced peak allocations separately for the selected forced layout/method;
exclude the caller-owned input and do not present this as process RSS/native RAM.

Functional gate: exact reconstruction for all candidates, strict bounds and
malformed-input rejection, unchanged frozen v1 vectors, package installation.
Performance result remains asset-specific even if byte-plane layout wins.
