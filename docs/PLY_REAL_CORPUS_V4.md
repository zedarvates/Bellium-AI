# Real polygonal corpus gate v4 — 18 September 2026

The v3 gate proved element-aware transposition on meshes this project generated
itself and recorded that no real polygonal PLY was available offline. This gate
measures an external, hash-pinned corpus, and closes the refusal those files
exposed.

## Corpus

Eight meshes from the `mikedh/trimesh` test corpus, pinned to commit
`fcf660feb0a14c68fd3945789e8ed77e260f9167`, with the repository's MIT licence read
from that same commit. Five are binary polygonal meshes and three are ASCII
polygonal meshes. `bunny.ply` and `suzanne.ply` were excluded because their
origins carry separate terms. The files live in a local evidence folder, are
never added to the repository or the wheel, and are recorded with URL, size and
SHA-256.

## Measured result

Seven of eight files are archived, every one byte for byte, three repetitions
each. Sizes are complete packets against a bare zlib of the same file.

| File | Source bytes | Bare zlib | Packet | Selected | Change vs zlib |
| --- | ---: | ---: | ---: | --- | ---: |
| cycloidal | 1,079,764 | 316,768 | 205,802 | byte-planes / delta | −35.0% |
| featuretype (ASCII) | 342,700 | 44,829 | 32,499 | **columns / zlib** | −27.5% |
| fixed_top | 109,902 | 38,367 | 33,904 | byte-planes / delta | −11.6% |
| octagonal_pocket | 62,502 | 22,900 | 22,348 | byte-planes / delta | −2.4% |
| mirror | 5,649 | 2,082 | 2,107 | byte-planes / zlib | +1.2% |
| sphere (ASCII) | 36,707 | 8,878 | 9,110 | opaque / zlib | +2.6% |
| tet | 303 | 173 | 394 | original / zlib | +127.7% |

The pattern of the earlier gates holds on third-party data: the archive wins on
the large meshes and loses on the small ones, where the fixed packet envelope of
a header, a metadata block and two SHA-256 digests dominates. `tet.ply` is 303
bytes of source; the codec is for assets, not for toy files. Automatic selection
also costs real time — 2.9 s to encode `octagonal_pocket` because it tries every
layout and method — so a caller who knows the mesh should force the pair.

The ASCII column layout earns its place on a real file: `featuretype`, exported
with canonical single-space rows, is 27.5% below a bare zlib, while
`sphere`, whose rows are irregularly spaced, is only archived verbatim.

## The refusal, and why it stands

`fuze_ascii.ply` is refused. The file declares, for its face element, an index
list, a texture-coordinate list and four colour scalars — 15 tokens per row — but
every one of its 1,000 face rows carries 21 tokens, ending in an undeclared `9`
followed by nine values. Its own header does not describe its own body.

This is a MeshLab export, not a synthetic case, and it is exactly the situation
where guessing would be worst: a codec that silently invented a third list would
produce an archive that decodes to something the original never was. The codec
refuses with an explicit message instead.

## What changed to make this possible

Multi-element ASCII. The header is preserved byte for byte and the body is read
line by line against each element's declared fields, including list arities. Any
ASCII body is stored verbatim, so irregular spacing, tabs, CRLF endings or a
missing final newline survive untouched. The column layout is offered only when
the body is canonical and every element has a uniform row width; an explicit
`columns` request on anything else fails rather than falling back silently. The
packet metadata carries a per-element `shape` of `[count, width]`, and the earlier
single-element `count`/`columns` shape still decodes.

A record whose list arity does not match the tokens present is refused, which is
what caught `fuze_ascii`.

## Limits

Eight third-party test meshes, not a population sample. The ultimate provenance
of each mesh was not re-verified beyond the repository licence, so the files stay
local and are not redistributed. Timings are single-machine Python and depend on
the linked zlib build, recorded beside the numbers. No `.fovea` comparison for
polygonal data, no Godot or Zig integration, no GPU path, no VR validation.

## Next gate

Decide the policy for headers that do not match their own body: refuse always
(today), or archive such files through an explicit escape hatch that records the
observed row shape without pretending to interpret it. Then re-run this table.
