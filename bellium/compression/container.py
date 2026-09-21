"""Bounded, self-contained BLCP v1 research packets; no pickle or side files."""

import hashlib
import json
import struct
import zlib

from .predictors import residuals

MAX_RAW_BYTES = 16 * 1024 * 1024
# Hard ceiling for callers that opt into a larger explicit limit. The wire
# format is unchanged; only the default guard moves, and it stays at 16 MiB.
ABSOLUTE_MAX_BYTES = 256 * 1024 * 1024
MAX_PREDICTIVE_BYTES = 64 * 1024
MAX_METADATA_BYTES = 4096
METHODS = ("raw", "zlib", "delta", "knn", "micro-nn")
# Magic, version, method, decoded bytes, metadata bytes, payload bytes, decoded SHA256.
HEADER = struct.Struct("<4sBBIII32s")


def _integer(value, name, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in {lower}..{upper}")
    return value


def _bytes(value, name):
    if not isinstance(value, bytes):
        raise ValueError(f"{name} must be bytes")
    return value


def _metadata(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate metadata key")
        result[key] = value
    return result


def pack(data: bytes, metadata: dict, *, method="auto", limit=MAX_RAW_BYTES) -> bytes:
    _bytes(data, "data")
    _integer(limit, "limit", 1, ABSOLUTE_MAX_BYTES)
    if len(data) > limit:
        raise ValueError("input exceeds the declared byte limit")
    if method not in (*METHODS, "auto"):
        raise ValueError("unknown compression method")
    if method in ("knn", "micro-nn") and len(data) > MAX_PREDICTIVE_BYTES:
        raise ValueError("predictive codecs are limited to 65536 bytes per packet")
    if type(metadata) is not dict or not isinstance(metadata.get("kind"), str):
        raise ValueError("metadata requires a kind")
    meta = json.dumps(metadata, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")
    if len(meta) > MAX_METADATA_BYTES:
        raise ValueError("metadata too large")
    candidates = METHODS if method == "auto" else (method,)
    best = None
    for name in candidates:
        if name in ("knn", "micro-nn") and len(data) > MAX_PREDICTIVE_BYTES:
            continue
        encoded = data if name in ("raw", "zlib") else residuals(data, name)
        payload = encoded if name == "raw" else zlib.compress(encoded, level=6)
        if best is None or len(payload) < len(best[1]):
            best = (name, payload)
    name, payload = best
    header = HEADER.pack(b"BLCP", 1, METHODS.index(name), len(data), len(meta),
                         len(payload), hashlib.sha256(data).digest())
    body = header + meta + payload
    return body + hashlib.sha256(body).digest()


def _parse(packet, max_output_bytes):
    _bytes(packet, "packet")
    _integer(max_output_bytes, "max_output_bytes", 0, ABSOLUTE_MAX_BYTES)
    if not HEADER.size + 32 <= len(packet) <= ABSOLUTE_MAX_BYTES + 16384:
        raise ValueError("invalid packet size")
    magic, version, code, size, meta_size, payload_size, digest = HEADER.unpack_from(packet)
    if magic != b"BLCP" or version != 1 or code >= len(METHODS):
        raise ValueError("unsupported packet version or method")
    if size > max_output_bytes or meta_size > MAX_METADATA_BYTES:
        raise ValueError("packet exceeds a declared size limit")
    method = METHODS[code]
    if method in ("knn", "micro-nn") and size > MAX_PREDICTIVE_BYTES:
        raise ValueError("predictive packet exceeds the work limit")
    end = HEADER.size + meta_size
    if end + payload_size + 32 != len(packet):
        raise ValueError("packet length mismatch")
    if hashlib.sha256(packet[:-32]).digest() != packet[-32:]:
        raise ValueError("packet checksum mismatch")
    try:
        meta = json.loads(packet[HEADER.size:end], object_pairs_hook=_metadata,
                          parse_constant=lambda _: (_ for _ in ()).throw(
                              ValueError("non-finite metadata")))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError("invalid packet metadata") from exc
    if type(meta) is not dict or not isinstance(meta.get("kind"), str):
        raise ValueError("packet metadata requires a kind")
    return method, size, digest, meta, packet[end:-32]


def inspect_packet(packet: bytes, *, max_output_bytes=MAX_RAW_BYTES) -> dict:
    """Validate the envelope and report sizes, without decompressing its payload."""
    method, size, _, meta, payload = _parse(packet, max_output_bytes)
    return {"format": "BLCP/1", "method": method, "raw_bytes": size,
            "packet_bytes": len(packet), "payload_bytes": len(payload),
            "saved_bytes": size - len(packet), "metadata": meta}


def unpack(packet: bytes, *, kind: str, max_output_bytes=MAX_RAW_BYTES):
    method, size, digest, meta, payload = _parse(packet, max_output_bytes)
    if meta["kind"] != kind:
        raise ValueError("wrong packet kind")
    if method == "raw":
        data = payload
    else:
        try:
            inflater = zlib.decompressobj()
            data = inflater.decompress(payload, size + 1)
        except zlib.error as exc:
            raise ValueError("invalid compressed payload") from exc
        if (len(data) != size or not inflater.eof or inflater.unused_data
                or inflater.unconsumed_tail):
            raise ValueError("compressed payload has wrong length or trailing data")
        if method in ("delta", "knn", "micro-nn"):
            data = residuals(data, method, decode=True)
    if len(data) != size or hashlib.sha256(data).digest() != digest:
        raise ValueError("decoded bytes failed integrity check")
    return data, meta
