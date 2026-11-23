"""A shared-memory channel so the 3D viewer can be updated without relaunching.

The viewer receives its arrays by value when the process is spawned, so the
only way to show a new depth map was to terminate the viewer and start another
one 400ms later.  That throws away the camera position, flashes a window, and
races: the old process may still hold the GL context when the new one asks for
it.

The viewer already polls ``shared`` every 50ms and rebuilds its geometry when a
value changes, so the missing piece was never the trigger - it was getting the
pixels across.  This provides that: one shared buffer per array, written in
place by the parent and mapped read-only by the viewer, with a generation
counter the existing poll can watch.

Shape is fixed for the life of the channel.  A crop changes the shape, and that
still needs a relaunch - correctly, since the mesh, UVs and calibration all
change with it.
"""
import numpy as np

try:
    from multiprocessing import shared_memory as _shm
    HAVE_SHM = True
except Exception:                                    # pragma: no cover
    _shm = None
    HAVE_SHM = False


class DepthChannel:
    """Shared buffers for the arrays the viewer rebuilds its mesh from."""

    KEYS = ("d_full", "d_hp")

    def __init__(self, arrays: dict):
        if not HAVE_SHM:
            raise RuntimeError("shared_memory unavailable")
        self._blocks = {}
        self.spec = {}
        for k in self.KEYS:
            a = np.ascontiguousarray(arrays[k], dtype=np.float32)
            blk = _shm.SharedMemory(create=True, size=a.nbytes)
            np.ndarray(a.shape, dtype=a.dtype, buffer=blk.buf)[:] = a
            self._blocks[k] = blk
            self.spec[k] = (blk.name, a.shape, "float32")

    def write(self, arrays: dict) -> bool:
        """Overwrite the buffers in place.  False if a shape no longer matches."""
        for k in self.KEYS:
            a = np.ascontiguousarray(arrays[k], dtype=np.float32)
            if a.shape != self.spec[k][1]:
                return False
        for k in self.KEYS:
            a = np.ascontiguousarray(arrays[k], dtype=np.float32)
            np.ndarray(self.spec[k][1], dtype=np.float32,
                       buffer=self._blocks[k].buf)[:] = a
        return True

    def close(self) -> None:
        """Release the buffers.  Leaked blocks outlive the process on macOS."""
        for blk in self._blocks.values():
            try:
                blk.close()
            except Exception:
                pass
            try:
                blk.unlink()
            except Exception:
                pass
        self._blocks.clear()


def attach(spec: dict) -> tuple[dict, list]:
    """Map the buffers described by *spec* read-only.  Viewer side.

    Returns (arrays, handles).  Keep *handles* alive for as long as the arrays
    are used: closing a SharedMemory invalidates every view onto it.
    """
    arrays, handles = {}, []
    for k, (name, shape, dtype) in spec.items():
        blk = _shm.SharedMemory(name=name)
        handles.append(blk)
        arrays[k] = np.ndarray(tuple(shape), dtype=np.dtype(dtype), buffer=blk.buf)
    return arrays, handles
