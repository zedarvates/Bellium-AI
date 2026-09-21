Model card: bellium/hybrid/normal-map-convention:v0

- Task: read what a normal map states about its own encoding, and convert the declared
  tangent-space handedness.
- Authority: consultative and inert. written_files, certified and pixels_changed stay false, false and
  0: the converted pixels are returned as data and nothing is written back.
- Inputs: image (the shared RGB integer contract), optional mask, from_convention and to_convention,
  each of which must be one of +y and -y. Undocumented query keys are refused rather than ignored.
- Outputs: the verdict, the mean Z, the negative-Z ratio, the mean unit-length deviation, the
  saturation ratio, the handedness detection with its margin, and the converted image or null.
- Statuses: inspected when no target is declared, converted when the target differs, unchanged when
  it does not, and abstain when the input is not a normal map or holds no normal pixel.
- The detection: both readings of the picture are decoded and their cell curls compared. The reading
  with the lower curl is the one that is a gradient of a surface; the other one is not a gradient at
  all unless the x slope is constant along the image vertical. It reads only the two channels that
  carry the slopes, so an inverted blue channel does not change the answer, and it needs no pixel
  scale because it compares two curls of the same field.
- Measured: 32 readings over four geometries at four noise levels, both handedness each time:
  8 decided, 24 abstained, 0 decided wrongly. The clean separations are 3.18 for a sphere and 5.13
  for a wave. An exact tilted plane is genuinely ambiguous (both readings measure 0.000 of curl and
  both are real surfaces), and a cone at rest separates by 6 times while the wrong reading stays
  below the sampling floor, so it abstains instead of claiming a decision.
- Symptoms: tangent-space, inverted-z, object-space, not-a-normal-map and ambiguous, measured on the
  controlled variants; a height map in RGB is named not-a-normal-map from its 0.442 mean unit-length
  deviation, and an object-space map from the half of its pixels whose Z is negative.
- Warnings: object_space_input, inverted_z_input, and
  converting_against_the_detected_convention when the declared target contradicts the evidence. The
  encoding is a declaration: the specialist converts what the caller declares and never rewrites an
  asset on the strength of a detection.
- Round trip: benchmarked at 0.084 to 0.249 degrees mean and a worst case of 0.691 degrees with
  Z reconstructed, 3.79 degrees at the silhouette once the normals are perturbed, and 40.9 to
  45.0 degrees if the map is stored as sRGB colour and read back as data.
- Limits: controlled 32 x 32 fields only. Object-space to tangent-space conversion, the UV basis,
  mirrored UVs, mipmap-safe filtering and block compression are not implemented; a converted picture
  is never written to disk.
- Evidence: tests/test_normal_map_io.py, scripts/benchmark_normal_map_io.py,
  docs/NORMAL_MAP_IO_GATE.md.

