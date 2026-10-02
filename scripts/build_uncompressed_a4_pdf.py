#!/usr/bin/env python3
"""Build an A4 PDF with raw RGB/RGBA PNG image streams and no PDF filters."""

import argparse
import json
import struct
import sys
import zlib
from pathlib import Path


A4_W = 595.275590551
A4_H = 841.88976378
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    return a if pa <= pb and pa <= pc else b if pb <= pc else c


def read_png(path):
    data = path.read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError(f"{path.name}: expected PNG")
    pos = len(PNG_SIGNATURE)
    width = height = bit_depth = color_type = interlace = None
    compressed = bytearray()
    while pos < len(data):
        if pos + 12 > len(data):
            raise ValueError(f"{path.name}: truncated PNG chunk")
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        kind = data[pos + 4 : pos + 8]
        chunk = data[pos + 8 : pos + 8 + length]
        pos += length + 12
        if kind == b"IHDR":
            width, height, bit_depth, color_type, compression, filt, interlace = struct.unpack(">IIBBBBB", chunk)
            if compression != 0 or filt != 0 or interlace != 0:
                raise ValueError(f"{path.name}: only non-interlaced standard PNG is supported")
            if bit_depth != 8 or color_type not in (2, 6):
                raise ValueError(f"{path.name}: only 8-bit RGB/RGBA PNG is supported")
        elif kind == b"IDAT":
            compressed.extend(chunk)
        elif kind == b"IEND":
            break
    if not width or not height or not compressed:
        raise ValueError(f"{path.name}: missing PNG image data")
    channels = 3 if color_type == 2 else 4
    stride = width * channels
    decoded = zlib.decompress(bytes(compressed))
    expected = (stride + 1) * height
    if len(decoded) != expected:
        raise ValueError(f"{path.name}: unexpected decompressed PNG length")
    rows = []
    previous = bytearray(stride)
    cursor = 0
    for _ in range(height):
        filter_type = decoded[cursor]
        source = decoded[cursor + 1 : cursor + 1 + stride]
        cursor += stride + 1
        row = bytearray(stride)
        for i, value in enumerate(source):
            left = row[i - channels] if i >= channels else 0
            up = previous[i]
            upper_left = previous[i - channels] if i >= channels else 0
            if filter_type == 0:
                row[i] = value
            elif filter_type == 1:
                row[i] = (value + left) & 255
            elif filter_type == 2:
                row[i] = (value + up) & 255
            elif filter_type == 3:
                row[i] = (value + ((left + up) // 2)) & 255
            elif filter_type == 4:
                row[i] = (value + paeth(left, up, upper_left)) & 255
            else:
                raise ValueError(f"{path.name}: unsupported PNG filter {filter_type}")
        rows.append(bytes(row))
        previous = row
    raw = b"".join(rows)
    if channels == 3:
        return width, height, raw, None
    rgb = bytearray(width * height * 3)
    alpha = bytearray(width * height)
    for i in range(width * height):
        rgb[i * 3 : i * 3 + 3] = raw[i * 4 : i * 4 + 3]
        alpha[i] = raw[i * 4 + 3]
    return width, height, bytes(rgb), bytes(alpha)


class PdfWriter:
    def __init__(self, path):
        self.stream = path.open("wb")
        self.offsets = [0]
        self.stream.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

    def object(self, body):
        object_id = len(self.offsets)
        self.offsets.append(self.stream.tell())
        self.stream.write(f"{object_id} 0 obj\n".encode("ascii"))
        self.stream.write(body)
        self.stream.write(b"\nendobj\n")
        return object_id

    def finish(self, root_id):
        startxref = self.stream.tell()
        count = len(self.offsets)
        self.stream.write(f"xref\n0 {count}\n".encode("ascii"))
        self.stream.write(b"0000000000 65535 f \n")
        for offset in self.offsets[1:]:
            self.stream.write(f"{offset:010d} 00000 n \n".encode("ascii"))
        self.stream.write(f"trailer\n<< /Size {count} /Root {root_id} 0 R >>\nstartxref\n{startxref}\n%%EOF\n".encode("ascii"))
        self.stream.close()


def stream_object(writer, dictionary, payload):
    return writer.object(dictionary + f" /Length {len(payload)} >>\nstream\n".encode("ascii") + payload + b"\nendstream")


def read_order(order_file, input_dir):
    if not order_file:
        return sorted(input_dir.glob("*.png"), key=lambda p: p.name.lower())
    raw = json.loads(order_file.read_text(encoding="utf-8"))
    indexed_plan = False
    if isinstance(raw, dict):
        values = raw.get("pages")
        if values is None:
            values = raw.get("output_plan")
            indexed_plan = values is not None
        if values is None:
            raise ValueError("order file must contain a pages or output_plan list")
    else:
        values = raw
    if not isinstance(values, list):
        raise ValueError("order file must be a JSON list or an object with a pages list")
    if indexed_plan:
        if not all(isinstance(item, dict) and "global_output_index" in item for item in values):
            raise ValueError("each output_plan entry needs a global_output_index")
        values = sorted(values, key=lambda item: int(item["global_output_index"]))
    paths = []
    for item in values:
        if isinstance(item, str):
            name = item
        elif isinstance(item, dict):
            name = item.get("final_image") or item.get("file") or item.get("path")
        else:
            raise ValueError("each order entry must be a filename or an object")
        if not name:
            raise ValueError("each order entry needs a PNG filename or final_image")
        path = Path(name)
        if not path.is_absolute():
            path = input_dir / path
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--order-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = read_order(args.order_file, args.input_dir)
    if not paths:
        raise ValueError("no PNG pages found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rebuild(args.output, paths)
    print(f"wrote {args.output} ({len(paths)} pages, raw RGB/RGBA streams)")


def rebuild(output, paths):
    # Rebuild in dependency order so every page can reference the known page tree id.
    if output.exists():
        output.unlink()
    writer = PdfWriter(output)
    page_ids = []
    image_specs = []
    for path in paths:
        width, height, rgb, alpha = read_png(path)
        mask_id = None
        if alpha is not None:
            mask_id = stream_object(writer, f"<< /Type /XObject /Subtype /Image /Width {width} /Height {height} /ColorSpace /DeviceGray /BitsPerComponent 8".encode("ascii"), alpha)
        image_dict = f"<< /Type /XObject /Subtype /Image /Width {width} /Height {height} /ColorSpace /DeviceRGB /BitsPerComponent 8".encode("ascii")
        if mask_id:
            image_dict += f" /SMask {mask_id} 0 R".encode("ascii")
        image_id = stream_object(writer, image_dict, rgb)
        scale = min(A4_W / width, A4_H / height)
        draw_w, draw_h = width * scale, height * scale
        x, y = (A4_W - draw_w) / 2, (A4_H - draw_h) / 2
        content = f"q\n{draw_w:.6f} 0 0 {draw_h:.6f} {x:.6f} {y:.6f} cm\n/Im0 Do\nQ\n".encode("ascii")
        content_id = stream_object(writer, b"<<", content)
        image_specs.append((width, height, image_id, content_id))
    page_tree_id = len(writer.offsets) + len(paths)
    for width, height, image_id, content_id in image_specs:
        page_ids.append(writer.object(f"<< /Type /Page /Parent {page_tree_id} 0 R /MediaBox [0 0 {A4_W:.6f} {A4_H:.6f}] /Resources << /XObject << /Im0 {image_id} 0 R >> >> /Contents {content_id} 0 R >>".encode("ascii")))
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    pages_id = writer.object(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("ascii"))
    assert pages_id == page_tree_id
    catalog_id = writer.object(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode("ascii"))
    writer.finish(catalog_id)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
