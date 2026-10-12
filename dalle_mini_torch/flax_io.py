"""Read Flax ``.msgpack`` checkpoints without JAX, Flax or the ``msgpack`` package.

A Flax checkpoint is a msgpack-encoded tree of dicts whose leaves are numpy arrays,
stored as msgpack extension type 1 holding ``(shape, dtype_name, raw_bytes)``.
The file is memory-mapped and the arrays are returned as views into it, so reading a
multi-gigabyte checkpoint costs no RAM until a tensor is actually copied out.

The format is plain data (no pickle), so loading an untrusted checkpoint cannot run code.
"""

import mmap
import struct

import numpy as np

# flax.serialization._MsgpackExtType
_EXT_NDARRAY = 1
_EXT_NATIVE_COMPLEX = 2
_EXT_NPSCALAR = 3

_CHUNKED_KEY = "__msgpack_chunked_array__"


class FlaxCheckpointError(ValueError):
    pass


class _Reader:
    def __init__(self, buf, pos=0, end=None):
        self.buf = buf
        self.pos = pos
        self.end = len(buf) if end is None else end

    def _take(self, n):
        start = self.pos
        if start + n > self.end:
            raise FlaxCheckpointError("truncated msgpack data")
        self.pos = start + n
        return start

    def _unpack(self, fmt):
        return struct.unpack_from(fmt, self.buf, self._take(struct.calcsize(fmt)))[0]

    def _str(self, n):
        start = self._take(n)
        return bytes(self.buf[start:start + n]).decode("utf-8")

    def _bin(self, n):
        # (offset, length) instead of bytes: array payloads are never copied here
        return (self._take(n), n)

    def _array(self, n):
        return [self.read() for _ in range(n)]

    def _map(self, n):
        out = {}
        for _ in range(n):
            key = self.read()
            out[key] = self.read()
        return out

    def _ext(self, n):
        code = self._unpack(">b")
        start = self._take(n)
        if code in (_EXT_NDARRAY, _EXT_NPSCALAR):
            shape, dtype_name, (offset, length) = _Reader(self.buf, start, start + n).read()
            dtype = np.dtype(dtype_name)
            count = int(np.prod(shape, dtype=np.int64)) if len(shape) else 1
            if count * dtype.itemsize != length:
                raise FlaxCheckpointError(f"array payload size mismatch for shape {shape} {dtype_name}")
            arr = np.frombuffer(self.buf, dtype=dtype, count=count, offset=offset).reshape(shape)
            return arr if code == _EXT_NDARRAY else arr[()]
        if code == _EXT_NATIVE_COMPLEX:
            real, imag = _Reader(self.buf, start, start + n).read()
            return complex(real, imag)
        raise FlaxCheckpointError(f"unsupported msgpack extension type {code}")

    def read(self):
        b = self.buf[self._take(1)]
        if b <= 0x7F:
            return b
        if b <= 0x8F:
            return self._map(b & 0x0F)
        if b <= 0x9F:
            return self._array(b & 0x0F)
        if b <= 0xBF:
            return self._str(b & 0x1F)
        if b >= 0xE0:
            return b - 0x100
        if b == 0xC0:
            return None
        if b == 0xC2:
            return False
        if b == 0xC3:
            return True
        if b == 0xC4:
            return self._bin(self._unpack(">B"))
        if b == 0xC5:
            return self._bin(self._unpack(">H"))
        if b == 0xC6:
            return self._bin(self._unpack(">I"))
        if b == 0xC7:
            return self._ext(self._unpack(">B"))
        if b == 0xC8:
            return self._ext(self._unpack(">H"))
        if b == 0xC9:
            return self._ext(self._unpack(">I"))
        if b == 0xCA:
            return self._unpack(">f")
        if b == 0xCB:
            return self._unpack(">d")
        if b == 0xCC:
            return self._unpack(">B")
        if b == 0xCD:
            return self._unpack(">H")
        if b == 0xCE:
            return self._unpack(">I")
        if b == 0xCF:
            return self._unpack(">Q")
        if b == 0xD0:
            return self._unpack(">b")
        if b == 0xD1:
            return self._unpack(">h")
        if b == 0xD2:
            return self._unpack(">i")
        if b == 0xD3:
            return self._unpack(">q")
        if 0xD4 <= b <= 0xD8:
            return self._ext(1 << (b - 0xD4))
        if b == 0xD9:
            return self._str(self._unpack(">B"))
        if b == 0xDA:
            return self._str(self._unpack(">H"))
        if b == 0xDB:
            return self._str(self._unpack(">I"))
        if b == 0xDC:
            return self._array(self._unpack(">H"))
        if b == 0xDD:
            return self._array(self._unpack(">I"))
        if b == 0xDE:
            return self._map(self._unpack(">H"))
        if b == 0xDF:
            return self._map(self._unpack(">I"))
        raise FlaxCheckpointError(f"unsupported msgpack type byte 0x{b:02x}")


def _unchunk(tree):
    """Undo flax.serialization's splitting of arrays larger than 1 GiB."""
    if not isinstance(tree, dict):
        return tree
    if tree.get(_CHUNKED_KEY):
        chunks = tree["chunks"]
        flat = np.concatenate([chunks[str(i)].reshape(-1) for i in range(len(chunks))])
        return flat.reshape(tree["shape"])
    return {k: _unchunk(v) for k, v in tree.items()}


def flatten(tree, prefix=()):
    """Flatten a nested param dict into ``{(path, ...): array}``."""
    out = {}
    for key, value in tree.items():
        path = prefix + (key,)
        if isinstance(value, dict):
            out.update(flatten(value, path))
        else:
            out[path] = value
    return out


class FlaxCheckpoint:
    """Memory-mapped Flax checkpoint. Use as a context manager; arrays are only valid while it is open."""

    def __init__(self, path):
        self.path = path
        self._file = open(path, "rb")
        try:
            # ACCESS_COPY gives writable copy-on-write pages, so torch.from_numpy accepts the views
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_COPY)
        except Exception:
            self._file.close()
            raise
        try:
            reader = _Reader(self._mmap)
            tree = reader.read()
            if not isinstance(tree, dict):
                raise FlaxCheckpointError(f"{path}: expected a dict of parameters at the top level")
            self.params = _unchunk(tree)
        except Exception:
            self.close()
            raise

    def flat(self):
        return flatten(self.params)

    def close(self):
        # numpy views keep the mmap alive, so drop them before closing
        self.params = None
        try:
            self._mmap.close()
        except BufferError:
            pass  # a caller still holds an array view; the mapping is released when it goes away
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
