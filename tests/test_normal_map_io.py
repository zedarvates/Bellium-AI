"""Normal-map pixels: the encoding, the decidable symptoms and the handedness."""

import math

import pytest

from bellium.contracts.schema import AuthorityMode
from bellium.material.controlled import perturbed_normals
from bellium.material.integration import gradients, integrability
from bellium.material.normals_io import (
    CONVENTIONS,
    NEAR_FLAT_PIXEL,
    decode_normal_map,
    detect_convention,
    encode_normal_map,
    flip_normal_convention,
    inspect_normal_map,
    quantize_height,
)
from bellium.material.photometric import normal_error, synthetic_geometry
from bellium.specialists.normal_map_io import convert_normal_map

SIZE = 32


def _field(geometry, sigma=0.0, seed=4):
    normals, mask = synthetic_geometry(geometry, size=SIZE)
    return perturbed_normals(normals, sigma, seed=seed), mask


def _bump(size=SIZE, strength=8.0):
    center = (size - 1) / 2.0
    radius = size * 0.45
    rows = []
    for r in range(size):
        row = []
        for c in range(size):
            x = (c - center) / radius
            y = (r - center) / radius
            kernel = math.exp(-(x * x + y * y) * 4.0)
            dzdx = -strength * x * kernel
            dzdy = -strength * y * kernel
            length = math.sqrt(dzdx * dzdx + dzdy * dzdy + 1.0)
            row.append((-dzdx / length, -dzdy / length, 1.0 / length))
        rows.append(row)
    return rows, [[1.0] * size for _ in range(size)]


def test_the_round_trip_stays_within_a_degree_on_a_clean_field() -> None:
    for geometry in ("sphere", "cone", "waves", "tilted-plane"):
        normals, mask = _field(geometry)
        image = encode_normal_map(normals)
        decoded, report = decode_normal_map(image, mask=mask)
        error = normal_error(decoded, normals, mask)
        assert report["convention"] == "+y"
        assert error["mean_deg"] < 0.35
        assert error["max_deg"] < 1.0
        assert report["unreconstructable"] == 0


def test_a_missing_normal_is_stored_as_a_flat_pixel() -> None:
    normals = [[None, (0.0, 0.0, 1.0)]]
    image = encode_normal_map(normals)
    assert image[0][0] == NEAR_FLAT_PIXEL
    decoded, _ = decode_normal_map(image)
    flat = decoded[0][0]
    assert flat[2] > 0.999
    assert abs(flat[0]) < 0.01 and abs(flat[1]) < 0.01


def test_the_mask_keeps_the_flat_pixels_out_of_the_statistics() -> None:
    normals, mask = _field("sphere")
    image = encode_normal_map(normals)
    unmasked, unmasked_report = decode_normal_map(image)
    masked, masked_report = decode_normal_map(image, mask=mask)
    assert unmasked_report["pixels"] == SIZE * SIZE
    assert masked_report["pixels"] == 648
    # the flat pixel carries Z = 1, so leaving it in pushes the mean up
    assert unmasked_report["mean_z"] > masked_report["mean_z"]
    assert unmasked[0][0] is not None and masked[0][0] is None
    with pytest.raises(ValueError, match="mask must share the image shape"):
        decode_normal_map(image, mask=[[1.0]])


def test_encoding_normalizes_before_it_quantizes() -> None:
    unit = encode_normal_map([[(0.0, 0.0, 1.0)]])
    long = encode_normal_map([[(0.0, 0.0, 25.0)]])
    assert unit == long


def test_the_flip_is_its_own_inverse_and_touches_one_channel() -> None:
    normals, _ = _field("waves")
    image = encode_normal_map(normals)
    flipped = flip_normal_convention(image)
    assert flip_normal_convention(flipped) == image
    assert flipped[3][5][0] == image[3][5][0]
    assert flipped[3][5][2] == image[3][5][2]
    assert flipped[3][5][1] == 255 - image[3][5][1]


def test_the_wrong_reading_of_a_bent_surface_is_not_a_gradient_field() -> None:
    """The measurement behind the detector: the flip alone is not a mirror."""
    normals, mask = _field("sphere")
    picture = encode_normal_map(normals, convention="+y")
    curls = {}
    for declaration in CONVENTIONS:
        decoded, _ = decode_normal_map(picture, convention=declaration, mask=mask)
        slope_x, slope_y, valid = gradients(decoded, mask)
        curls[declaration] = integrability(slope_x, slope_y, valid)["curl_ratio"]
    assert curls["+y"] < 0.15
    assert curls["-y"] > 0.15
    found = detect_convention(picture, mask=mask)
    assert found["status"] == "decided"
    assert found["convention"] == "+y"
    assert found["margin"] > 2.0


def test_detection_decides_both_handedness_of_a_surface_that_bends() -> None:
    for label, normals, mask in (
        ("sphere", *_field("sphere")),
        ("waves", *_field("waves")),
        ("bump", *_bump()),
    ):
        for truth in CONVENTIONS:
            picture = encode_normal_map(normals, convention=truth)
            found = detect_convention(picture, mask=mask)
            assert found["status"] == "decided", label
            assert found["convention"] == truth, label
            assert found["margin"] >= 2.0, label


def test_detection_abstains_when_both_readings_are_integrable() -> None:
    normals, mask = _field("tilted-plane")
    picture = encode_normal_map(normals)
    found = detect_convention(picture, mask=mask)
    assert found["status"] == "abstain"
    assert found["reason"] == "both_readings_integrable"
    assert found["convention"] is None
    assert found["readings"]["+y"]["curl_ratio"] == 0.0
    assert found["readings"]["-y"]["curl_ratio"] == 0.0


def test_detection_abstains_when_noise_drowns_the_margin() -> None:
    normals, mask = _field("cone", sigma=0.1)
    picture = encode_normal_map(normals)
    found = detect_convention(picture, mask=mask)
    assert found["status"] == "abstain"
    assert found["reason"] == "margin_below_declared"
    assert found["margin"] < 2.0
    assert found["convention"] is None


def test_inspect_names_the_symptoms_a_map_can_state() -> None:
    normals, mask = _field("sphere")
    tangent = encode_normal_map(normals)
    assert inspect_normal_map(tangent, mask=mask)["verdict"] == "tangent-space"
    inverted = encode_normal_map(
        [[(n[0], n[1], -n[2]) if n else None for n in row] for row in normals]
    )
    assert inspect_normal_map(inverted, mask=mask)["verdict"] == "inverted-z"
    object_space = [
        [
            (
                math.sin(math.pi * (c + 0.5) / SIZE) * math.cos(2 * math.pi * (r + 0.5) / SIZE),
                math.sin(math.pi * (c + 0.5) / SIZE) * math.sin(2 * math.pi * (r + 0.5) / SIZE),
                math.cos(math.pi * (c + 0.5) / SIZE),
            )
            for c in range(SIZE)
        ]
        for r in range(SIZE)
    ]
    reported = inspect_normal_map(encode_normal_map(object_space))
    assert reported["verdict"] == "object-space"
    assert reported["negative_z_ratio"] > 0.2
    grey = [
        [(level * 7, level * 7, level * 7) for level in range(SIZE)]
        for _ in range(SIZE)
    ]
    assert inspect_normal_map(grey)["verdict"] == "not-a-normal-map"


def test_the_handedness_is_reported_as_a_measurement_not_a_constant() -> None:
    normals, mask = _field("sphere")
    report = inspect_normal_map(encode_normal_map(normals), mask=mask)
    assert report["y_convention_decidable"] is True
    assert report["y_convention_detection"]["convention"] == "+y"
    plain, plane_mask = _field("tilted-plane")
    undecided = inspect_normal_map(encode_normal_map(plain), mask=plane_mask)
    assert undecided["y_convention_decidable"] is False
    assert undecided["y_convention_detection"]["reason"] == "both_readings_integrable"


def test_integrability_separates_a_ramp_from_a_rotational_field() -> None:
    normals, mask = _field("tilted-plane")
    slope_x, slope_y, valid = gradients(normals, mask)
    clean = integrability(slope_x, slope_y, valid)
    assert clean["curl_ratio"] == 0.0
    assert clean["verdict"] == "integrable"
    turned_x = [[slope_x[r][c] - 0.6 * (r - 15.5) if valid[r][c] else 0.0 for c in range(SIZE)]
                for r in range(SIZE)]
    turned_y = [[slope_y[r][c] + 0.6 * (c - 15.5) if valid[r][c] else 0.0 for c in range(SIZE)]
                for r in range(SIZE)]
    turned = integrability(turned_x, turned_y, valid)
    # a rotation saturates near the ceiling whatever its strength: the ratio
    # divides by a slope magnitude that grows with the rotation itself
    assert 0.15 < turned["curl_ratio"] <= 0.25
    assert turned["verdict"] == "suspect"
    assert turned["curl_ratio"] > clean["curl_ratio"]
    assert turned["cells"] == clean["cells"] == (SIZE - 1) ** 2


def test_integrability_flags_a_field_that_is_not_a_gradient() -> None:
    """A slope that alternates sign every row is not the gradient of anything."""
    alternating = [[0.1 if r % 2 else -0.1 for _ in range(SIZE)] for r in range(SIZE)]
    flat = [[0.0] * SIZE for _ in range(SIZE)]
    valid = [[1.0] * SIZE for _ in range(SIZE)]
    report = integrability(alternating, flat, valid)
    assert report["curl_ratio"] > 0.25
    assert report["verdict"] == "not_integrable"


def test_integrability_reports_unknown_without_an_interior_pixel() -> None:
    report = integrability([[1.0]], [[1.0]], [[1.0]])
    assert report["curl_ratio"] is None
    assert report["verdict"] == "unknown"
    assert report["cells"] == 0


def test_quantize_height_states_what_the_quantization_cost() -> None:
    normals, mask = _field("sphere")
    from bellium.material.integration import integrate_height

    result = integrate_height(normals, mask, method="cumulative-average")
    eight = quantize_height(result["height"], result["valid"], bit_depth=8)
    sixteen = quantize_height(result["height"], result["valid"], bit_depth=16)
    assert eight["pixels"] == 648
    assert isinstance(eight["values"][16][16], int)
    assert 0 <= max(max(row) for row in eight["values"]) <= 255
    assert eight["worst_error"] <= eight["step"] / 2.0 + 1e-9
    assert sixteen["step"] < eight["step"] / 200.0
    assert sixteen["worst_error"] < eight["worst_error"] / 200.0
    assert eight["worst_error"] / (eight["span"][1] - eight["span"][0]) < 0.002


def test_quantize_height_refuses_a_bad_depth_or_shape() -> None:
    height = [[0.0, 1.0], [0.5, 0.25]]
    valid = [[1.0, 1.0], [1.0, 0.0]]
    for depth in (0, 17, True, 8.0, "high"):
        with pytest.raises(ValueError, match="bit_depth must be"):
            quantize_height(height, valid, bit_depth=depth)
    with pytest.raises(ValueError, match="mask must share the height shape"):
        quantize_height(height, [[1.0] * 3], bit_depth=8)
    with pytest.raises(ValueError, match="span must not be inverted"):
        quantize_height(height, valid, span=(1.0, 0.0))


def test_the_specialist_inspects_without_converting() -> None:
    normals, mask = _field("sphere")
    result = convert_normal_map({"image": encode_normal_map(normals), "mask": mask})
    assert result.authority_mode is AuthorityMode.CONSULTATIVE
    output = result.output
    assert output["status"] == "inspected"
    assert output["converted"] is None
    assert output["written_files"] is False
    assert output["certified"] is False
    assert output["pixels_changed"] == 0
    assert output["y_convention_detection"]["convention"] == "+y"
    assert result.confidence == 0.8


def test_the_specialist_converts_only_the_declared_flip() -> None:
    normals, mask = _field("waves")
    image = encode_normal_map(normals)
    result = convert_normal_map(
        {"image": image, "from_convention": "+y", "to_convention": "-y"}
    )
    assert result.output["status"] == "converted"
    assert result.output["converted"] == flip_normal_convention(image)
    assert "converting_against_the_detected_convention" in result.output["warnings"]
    same = convert_normal_map({"image": image, "from_convention": "+y", "to_convention": "+y"})
    assert same.output["status"] == "unchanged"
    assert same.output["warnings"] == []


def test_the_specialist_abstains_on_a_map_that_is_not_a_normal_map() -> None:
    grey = [[(index * 5, index * 5, index * 5) for index in range(SIZE)] for _ in range(2)]
    result = convert_normal_map({"image": grey})
    assert result.abstained is True
    assert result.confidence == 0.0
    assert result.output["reason"] == "not_a_normal_map"
    assert result.output["converted"] is None


def test_the_specialist_refuses_undocumented_keys_and_bad_conventions() -> None:
    image = encode_normal_map([[(0.0, 0.0, 1.0)]])
    with pytest.raises(ValueError, match="unknown query keys"):
        convert_normal_map({"image": image, "certify": True})
    with pytest.raises(ValueError, match="image must be provided"):
        convert_normal_map({})
    with pytest.raises(ValueError, match="query must be an object"):
        convert_normal_map(None)
    with pytest.raises(ValueError, match="from_convention must be"):
        convert_normal_map({"image": image, "from_convention": "opengl"})
    with pytest.raises(ValueError, match="to_convention must be"):
        convert_normal_map({"image": image, "to_convention": "directx"})


def test_pixels_and_channels_are_validated() -> None:
    # the shared image contract answers first, and it answers with its own words
    for broken in ([[(1, 2)]], [[(1, 2.5, 3)]], [[(256, 2, 3)]]):
        with pytest.raises(ValueError, match="three integer RGB channels"):
            decode_normal_map(broken)
    with pytest.raises(ValueError, match="convention must be one of"):
        decode_normal_map([[(1, 2, 3)]], convention="both")
    with pytest.raises(ValueError, match="normals are empty"):
        encode_normal_map([])
    with pytest.raises(ValueError, match="equal width"):
        encode_normal_map([[(0.0, 0.0, 1.0)], [(0.0, 0.0, 1.0), (0.0, 0.0, 1.0)]])
    with pytest.raises(ValueError, match="three components"):
        encode_normal_map([[(0.0, 1.0)]])
    with pytest.raises(ValueError, match="must be finite"):
        encode_normal_map([[(0.0, 0.0, float("inf"))]])
