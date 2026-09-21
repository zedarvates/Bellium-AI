from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

@dataclass(frozen=True)
class FastImageInfo:
    format: str
    width: int
    height: int
    channels: int
    bit_depth: int
    orientation: int
    is_animated: bool
    aspect_ratio: float
    bytes_read: int

def _parse_exif_orientation(payload: bytes) -> int:
    if len(payload) < 8 or not payload.startswith(b'Exif\x00\x00'):
        return 1
    tiff_header = payload[6:]
    if len(tiff_header) < 8:
        return 1
    endian_sig = tiff_header[:2]
    if endian_sig == b'II':
        endian = '<'
    elif endian_sig == b'MM':
        endian = '>'
    else:
        return 1
    tag_mark, offset = struct.unpack(f'{endian}HI', tiff_header[2:8])
    if tag_mark != 42 or offset >= len(tiff_header):
        return 1
    idx = offset
    if idx + 2 > len(tiff_header):
        return 1
    num_entries = struct.unpack(f'{endian}H', tiff_header[idx:idx + 2])[0]
    idx += 2
    for _ in range(num_entries):
        if idx + 12 > len(tiff_header):
            break
        tag, typ, count, val_or_off = struct.unpack(f'{endian}HHI4s', tiff_header[idx:idx + 12])
        idx += 12
        if tag == 0x0112:  # Orientation tag
            orient = struct.unpack(f'{endian}H', val_or_off[:2])[0]
            if 1 <= orient <= 8:
                return orient
    return 1

def fast_info(source: bytes | str | Path) -> FastImageInfo:
    if isinstance(source, (str, Path)):
        p = Path(source)
        with p.open('rb') as f:
            header = f.read(65536)
    else:
        header = bytes(source[:65536])
    
    n = len(header)
    if n < 4:
        raise ValueError('source has insufficient bytes to detect image format')
        
    # 1. PNG: 8-byte signature + IHDR chunk
    if header.startswith(b'\x89PNG\r\n\x1a\n'):
        if n < 24:
            raise ValueError('truncated PNG header')
        w, h = struct.unpack('>II', header[16:24])
        bit_depth = header[24]
        color_type = header[25]
        ch_map = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
        channels = ch_map.get(color_type, 3)
        is_animated = b'acTL' in header
        aspect = round(w / h, 4) if h > 0 else 1.0
        return FastImageInfo('PNG', w, h, channels, bit_depth, 1, is_animated, aspect, min(n, 1024))
        
    # 2. GIF: GIF87a or GIF89a
    if header.startswith(b'GIF87a') or header.startswith(b'GIF89a'):
        if n < 10:
            raise ValueError('truncated GIF header')
        w, h = struct.unpack('<HH', header[6:10])
        aspect = round(w / h, 4) if h > 0 else 1.0
        is_animated = b'\x21\xf9\x04' in header  # Graphic Control Extension
        return FastImageInfo('GIF', w, h, 3, 8, 1, is_animated, aspect, min(n, 1024))
        
    # 3. JPEG: 0xFFD8
    if header.startswith(b'\xff\xd8'):
        idx = 2
        orientation = 1
        w = h = channels = 0
        while idx + 4 <= n:
            if header[idx] != 0xff:
                idx += 1
                continue
            marker = header[idx + 1]
            if marker in (0xd9, 0xda):  # EOI or SOS (start of scan)
                break
            if marker == 0x00 or (0xd0 <= marker <= 0xd7):  # RST
                idx += 2
                continue
            seg_len = struct.unpack('>H', header[idx + 2:idx + 4])[0]
            if seg_len < 2:
                break
            seg_data = header[idx + 4:idx + 2 + seg_len]
            if marker == 0xe1 and seg_data.startswith(b'Exif\x00\x00'):
                orientation = _parse_exif_orientation(seg_data)
            elif marker in (0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf):
                # SOF marker: precision, height, width, components
                if len(seg_data) >= 6:
                    precision, h, w, channels = struct.unpack('>BHHB', seg_data[:6])
                    break
            idx += 2 + seg_len
        if w > 0 and h > 0:
            aspect = round(w / h, 4)
            return FastImageInfo('JPEG', w, h, channels or 3, 8, orientation, False, aspect, idx)
        raise ValueError('JPEG dimensions could not be found in stream')
        
    # 4. WebP: RIFF ... WEBP
    if header.startswith(b'RIFF') and n >= 16 and header[8:12] == b'WEBP':
        tag = header[12:16]
        if tag == b'VP8 ' and n >= 30:
            w, h = struct.unpack('<HH', header[26:30])
            w &= 0x3fff
            h &= 0x3fff
            aspect = round(w / h, 4) if h > 0 else 1.0
            return FastImageInfo('WEBP', w, h, 3, 8, 1, False, aspect, 30)
        elif tag == b'VP8L' and n >= 25:
            b0, b1, b2, b3 = struct.unpack('4B', header[21:25])
            w = 1 + (((b1 & 0x3f) << 8) | b0)
            h = 1 + (((b3 & 0x0f) << 10) | (b2 << 2) | ((b1 & 0xc0) >> 6))
            aspect = round(w / h, 4) if h > 0 else 1.0
            return FastImageInfo('WEBP', w, h, 4, 8, 1, False, aspect, 25)
        elif tag == b'VP8X' and n >= 30:
            flags = header[20]
            is_anim = bool(flags & 0x02)
            w = 1 + struct.unpack('<I', header[24:27] + b'\x00')[0]
            h = 1 + struct.unpack('<I', header[27:30] + b'\x00')[0]
            aspect = round(w / h, 4) if h > 0 else 1.0
            return FastImageInfo('WEBP', w, h, 4, 8, 1, is_anim, aspect, 30)
            
    # 5. BMP: BM
    if header.startswith(b'BM') and n >= 26:
        w, h = struct.unpack('<ii', header[18:26])
        h = abs(h)
        bpp = struct.unpack('<H', header[28:30])[0] if n >= 30 else 24
        aspect = round(w / h, 4) if h > 0 else 1.0
        return FastImageInfo('BMP', w, h, bpp // 8 or 3, 8, 1, False, aspect, 30)
        
    # 6. PPM: P6 or P3
    if header.startswith(b'P6') or header.startswith(b'P3'):
        tokens = []
        idx = 0
        while len(tokens) < 4 and idx < n:
            while idx < n and header[idx] in b' \t\r\n':
                idx += 1
            if idx >= n:
                break
            if header[idx] == ord(b'#'):
                while idx < n and header[idx] != ord(b'\n'):
                    idx += 1
                continue
            t_start = idx
            while idx < n and header[idx] not in b' \t\r\n#':
                idx += 1
            tokens.append(header[t_start:idx])
        if len(tokens) >= 3:
            w, h = int(tokens[1]), int(tokens[2])
            aspect = round(w / h, 4) if h > 0 else 1.0
            return FastImageInfo('PPM', w, h, 3, 8, 1, False, aspect, idx)
            
    # 7. TIFF: II or MM
    if (header.startswith(b'II*\x00') or header.startswith(b'MM\x00*')) and n >= 8:
        endian = '<' if header[:2] == b'II' else '>'
        ifd_offset = struct.unpack(f'{endian}I', header[4:8])[0]
        if ifd_offset + 2 <= n:
            num_entries = struct.unpack(f'{endian}H', header[ifd_offset:ifd_offset + 2])[0]
            idx = ifd_offset + 2
            w = h = 0
            orient = 1
            for _ in range(num_entries):
                if idx + 12 > n:
                    break
                tag, typ, cnt, val = struct.unpack(f'{endian}HHI4s', header[idx:idx + 12])
                idx += 12
                if tag == 256:  # ImageWidth
                    w = struct.unpack(f'{endian}H' if typ == 3 else f'{endian}I', val[:2 if typ == 3 else 4])[0]
                elif tag == 257:  # ImageLength
                    h = struct.unpack(f'{endian}H' if typ == 3 else f'{endian}I', val[:2 if typ == 3 else 4])[0]
                elif tag == 274:  # Orientation
                    orient = struct.unpack(f'{endian}H', val[:2])[0]
            if w > 0 and h > 0:
                aspect = round(w / h, 4)
                return FastImageInfo('TIFF', w, h, 3, 8, orient, False, aspect, idx)
                
    raise ValueError('unknown or unsupported image format for fast sniffing')
