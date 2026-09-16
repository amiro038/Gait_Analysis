"""Minimal reader for binary FBX files (versions 7100-7700).

Parses the raw node-record tree. No interpretation of FBX semantics here --
that lives in fbx_scene.py.
"""
import struct
import zlib


class Node:
    __slots__ = ("name", "props", "children")

    def __init__(self, name, props, children):
        self.name = name
        self.props = props
        self.children = children

    def find(self, name):
        for c in self.children:
            if c.name == name:
                return c
        return None

    def find_all(self, name):
        return [c for c in self.children if c.name == name]

    def __repr__(self):
        return f"<Node {self.name} props={len(self.props)} children={len(self.children)}>"


def _read_prop(buf, off):
    """Read one property. Returns (value, new_offset)."""
    t = buf[off:off + 1].decode("ascii")
    off += 1

    # --- scalars -----------------------------------------------------
    if t == "Y":
        return struct.unpack_from("<h", buf, off)[0], off + 2
    if t == "C":
        return bool(buf[off]), off + 1
    if t == "I":
        return struct.unpack_from("<i", buf, off)[0], off + 4
    if t == "F":
        return struct.unpack_from("<f", buf, off)[0], off + 4
    if t == "D":
        return struct.unpack_from("<d", buf, off)[0], off + 8
    if t == "L":
        return struct.unpack_from("<q", buf, off)[0], off + 8

    # --- arrays ------------------------------------------------------
    if t in "fdlib":
        n, enc, clen = struct.unpack_from("<III", buf, off)
        off += 12
        raw = buf[off:off + clen]
        off += clen
        if enc == 1:
            raw = zlib.decompress(raw)
        fmt = {"f": "f", "d": "d", "l": "q", "i": "i", "b": "b"}[t]
        vals = struct.unpack_from("<" + fmt * n, raw, 0)
        return list(vals), off

    # --- blobs -------------------------------------------------------
    if t in "SR":
        n = struct.unpack_from("<I", buf, off)[0]
        off += 4
        raw = buf[off:off + n]
        off += n
        if t == "S":
            return raw.decode("utf-8", errors="replace"), off
        return raw, off

    raise ValueError(f"Unknown FBX property type {t!r} at offset {off - 1}")


def _read_node(buf, off, version):
    """Read one node record. Returns (Node|None, new_offset). None = sentinel."""
    if version >= 7500:
        end_off, nprops, plen = struct.unpack_from("<QQQ", buf, off)
        off += 24
        sentinel_len = 25
    else:
        end_off, nprops, plen = struct.unpack_from("<III", buf, off)
        off += 12
        sentinel_len = 13

    name_len = buf[off]
    off += 1

    if end_off == 0:  # null record terminating a child list
        return None, off + name_len

    name = buf[off:off + name_len].decode("utf-8", errors="replace")
    off += name_len

    props = []
    for _ in range(nprops):
        v, off = _read_prop(buf, off)
        props.append(v)

    children = []
    # Nested list present only if there is space before end_off
    if off < end_off:
        while off < end_off - sentinel_len:
            child, off = _read_node(buf, off, version)
            if child is None:
                break
            children.append(child)
        off = end_off

    return Node(name, props, children), off


def load(path):
    """Parse an FBX file. Returns (root_node, version)."""
    with open(path, "rb") as fh:
        buf = fh.read()

    if buf[:21] != b"Kaydara FBX Binary  \x00":
        raise ValueError(f"{path} is not a binary FBX file (ASCII FBX unsupported)")

    version = struct.unpack_from("<I", buf, 23)[0]
    off = 27

    children = []
    n = len(buf)
    while off < n - 100:  # footer region
        node, off = _read_node(buf, off, version)
        if node is None:
            break
        children.append(node)

    return Node("__root__", [], children), version


def dump(node, depth=0, max_depth=3, max_children=12):
    """Print an indented summary of the node tree."""
    pad = "  " * depth
    if depth > 0:
        pv = []
        for p in node.props[:4]:
            if isinstance(p, list):
                pv.append(f"[{len(p)} vals]")
            elif isinstance(p, bytes):
                pv.append(f"<{len(p)} bytes>")
            else:
                pv.append(repr(p)[:40])
        print(f"{pad}{node.name}: {', '.join(pv)}")
    if depth >= max_depth:
        if node.children:
            print(f"{pad}  ... {len(node.children)} children")
        return
    for c in node.children[:max_children]:
        dump(c, depth + 1, max_depth, max_children)
    if len(node.children) > max_children:
        print(f"{pad}  ... +{len(node.children) - max_children} more children")
