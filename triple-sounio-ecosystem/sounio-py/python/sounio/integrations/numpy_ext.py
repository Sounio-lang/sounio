"""NumPy integration: UncertainArray — vectorised GUM uncertainty propagation.

Requires numpy >= 1.20.  Import guard is deliberately permissive so the rest
of the sounio package works without numpy installed.
"""

from __future__ import annotations

import json
from multiprocessing.shared_memory import SharedMemory
from typing import TYPE_CHECKING, List, Optional, Union

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _HAS_NUMPY = False
    np = None  # type: ignore[assignment]

if TYPE_CHECKING:
    import numpy as np  # noqa: F811

from ..knowledge import Knowledge


class UncertainArray:
    """A NumPy array paired with per-element GUM standard uncertainties.

    Parameters
    ----------
    values : array-like
        Central estimates.  Converted to float64 ndarray.
    epsilons : array-like, optional
        Per-element uncertainties.  Defaults to zeros.
    provenance : str
        Label for the dataset.

    Arithmetic
    ----------
    Addition / subtraction : ε = sqrt(εa² + εb²)  element-wise
    Multiplication         : ε = |a·b| · sqrt((εa/a)² + (εb/b)²)
    Division               : same relative-uncertainty rule
    Scalar operations      : uncertainty scales linearly with |factor|
    """

    def __init__(
        self,
        values,
        epsilons=None,
        provenance: str = "numpy",
    ) -> None:
        if not _HAS_NUMPY:
            raise ImportError("numpy is required for UncertainArray")
        self.values = np.asarray(values, dtype=np.float64)
        if epsilons is None:
            self.epsilons = np.zeros_like(self.values)
        else:
            self.epsilons = np.asarray(epsilons, dtype=np.float64)
        if self.values.shape != self.epsilons.shape:
            raise ValueError(
                f"values shape {self.values.shape} != epsilons shape {self.epsilons.shape}"
            )
        self.provenance = str(provenance)

    # ---- Arithmetic -------------------------------------------------------

    def __add__(self, other: Union["UncertainArray", float, int]) -> "UncertainArray":
        if isinstance(other, UncertainArray):
            return UncertainArray(
                self.values + other.values,
                np.sqrt(self.epsilons**2 + other.epsilons**2),
                f"({self.provenance})+({other.provenance})",
            )
        scalar = float(other)
        return UncertainArray(self.values + scalar, self.epsilons.copy(), self.provenance)

    def __radd__(self, other) -> "UncertainArray":
        return self.__add__(other)

    def __sub__(self, other: Union["UncertainArray", float, int]) -> "UncertainArray":
        if isinstance(other, UncertainArray):
            return UncertainArray(
                self.values - other.values,
                np.sqrt(self.epsilons**2 + other.epsilons**2),
                f"({self.provenance})-({other.provenance})",
            )
        scalar = float(other)
        return UncertainArray(self.values - scalar, self.epsilons.copy(), self.provenance)

    def __rsub__(self, other) -> "UncertainArray":
        scalar = float(other)
        return UncertainArray(scalar - self.values, self.epsilons.copy(), self.provenance)

    def __mul__(self, other: Union["UncertainArray", float, int]) -> "UncertainArray":
        if isinstance(other, UncertainArray):
            vals = self.values * other.values
            # Relative uncertainties — suppress divide-by-zero warnings
            with np.errstate(invalid="ignore", divide="ignore"):
                rel_a = np.where(self.values != 0, self.epsilons / np.abs(self.values), 0.0)
                rel_b = np.where(other.values != 0, other.epsilons / np.abs(other.values), 0.0)
            return UncertainArray(
                vals,
                np.abs(vals) * np.sqrt(rel_a**2 + rel_b**2),
                f"({self.provenance})*({other.provenance})",
            )
        factor = float(other)
        return UncertainArray(
            self.values * factor,
            self.epsilons * abs(factor),
            self.provenance,
        )

    def __rmul__(self, other) -> "UncertainArray":
        return self.__mul__(other)

    def __truediv__(self, other: Union["UncertainArray", float, int]) -> "UncertainArray":
        if isinstance(other, UncertainArray):
            with np.errstate(invalid="ignore", divide="ignore"):
                vals = self.values / other.values
                rel_a = np.where(self.values != 0, self.epsilons / np.abs(self.values), 0.0)
                rel_b = np.where(other.values != 0, other.epsilons / np.abs(other.values), 0.0)
            return UncertainArray(
                vals,
                np.abs(vals) * np.sqrt(rel_a**2 + rel_b**2),
                f"({self.provenance})/({other.provenance})",
            )
        divisor = float(other)
        if divisor == 0.0:
            raise ZeroDivisionError("UncertainArray division by zero scalar")
        return UncertainArray(
            self.values / divisor,
            self.epsilons / abs(divisor),
            self.provenance,
        )

    def __neg__(self) -> "UncertainArray":
        return UncertainArray(-self.values, self.epsilons.copy(), f"-({self.provenance})")

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, idx) -> Knowledge:
        return Knowledge(float(self.values[idx]), float(self.epsilons[idx]), self.provenance)

    # ---- Statistical summaries -------------------------------------------

    def mean(self) -> Knowledge:
        """Arithmetic mean with combined GUM uncertainty (ε / √n)."""
        n = len(self.values)
        if n == 0:
            return Knowledge(float("nan"), float("nan"), self.provenance)
        return Knowledge(
            float(np.mean(self.values)),
            float(np.sqrt(np.sum(self.epsilons**2))) / n,
            f"mean({self.provenance})",
        )

    def std(self) -> Knowledge:
        """Standard deviation with combined uncertainty estimate."""
        return Knowledge(
            float(np.std(self.values)),
            float(np.mean(self.epsilons)),
            f"std({self.provenance})",
        )

    def sum(self) -> Knowledge:
        """Element-wise sum with GUM-combined uncertainty."""
        return Knowledge(
            float(np.sum(self.values)),
            float(np.sqrt(np.sum(self.epsilons**2))),
            f"sum({self.provenance})",
        )

    def min(self) -> Knowledge:
        idx = int(np.argmin(self.values))
        return Knowledge(float(self.values[idx]), float(self.epsilons[idx]), self.provenance)

    def max(self) -> Knowledge:
        idx = int(np.argmax(self.values))
        return Knowledge(float(self.values[idx]), float(self.epsilons[idx]), self.provenance)

    # ---- Conversion -------------------------------------------------------

    def to_knowledge_list(self) -> List[Knowledge]:
        """Convert to a list of Knowledge objects."""
        return [
            Knowledge(float(v), float(e), self.provenance)
            for v, e in zip(self.values, self.epsilons)
        ]

    @classmethod
    def from_knowledge_list(cls, knowledges: List[Knowledge]) -> "UncertainArray":
        """Build an UncertainArray from a list of Knowledge objects."""
        if not knowledges:
            return cls([], [], "")
        values = [k.value for k in knowledges]
        epsilons = [k.epsilon for k in knowledges]
        prov = knowledges[0].provenance
        return cls(values, epsilons, prov)

    # ---- Shared memory ----------------------------------------------------

    def to_shared_memory(self) -> tuple:
        """Write values and epsilons to POSIX shared memory blocks (zero-copy IPC).

        Both blocks are created as new named segments. The caller is responsible
        for releasing them when done by calling
        :meth:`UncertainArray.free_shared_memory`.

        Returns
        -------
        tuple of (values_shm_name, epsilons_shm_name, meta_json)
            Pass these three strings to :meth:`from_shared_memory` in another
            process to reconstruct the array without copying data through the OS.

        Examples
        --------
        >>> a = UncertainArray([1.0, 2.0, 3.0], [0.1, 0.1, 0.1], "sensor")
        >>> names = a.to_shared_memory()
        >>> b = UncertainArray.from_shared_memory(*names)
        >>> UncertainArray.free_shared_memory(*names[:2])
        """
        if not _HAS_NUMPY:
            raise ImportError("numpy is required for shared memory support")

        # Ensure contiguous float64 arrays so nbytes is reliable.
        values = np.ascontiguousarray(self.values, dtype=np.float64)
        epsilons = np.ascontiguousarray(self.epsilons, dtype=np.float64)

        # Guard against zero-byte segments (SharedMemory requires size > 0).
        v_shm = SharedMemory(create=True, size=max(values.nbytes, 8))
        e_shm = SharedMemory(create=True, size=max(epsilons.nbytes, 8))

        v_buf = np.ndarray(values.shape, dtype=np.float64, buffer=v_shm.buf)
        e_buf = np.ndarray(epsilons.shape, dtype=np.float64, buffer=e_shm.buf)
        v_buf[:] = values
        e_buf[:] = epsilons

        meta = json.dumps({
            "shape": list(values.shape),
            "dtype": "float64",
            "provenance": self.provenance,
        })

        v_name = v_shm.name
        e_name = e_shm.name

        # Close local handles — the OS keeps the segments alive until unlinked.
        v_shm.close()
        e_shm.close()

        return v_name, e_name, meta

    @classmethod
    def from_shared_memory(
        cls,
        values_shm_name: str,
        epsilons_shm_name: str,
        meta_json: str,
    ) -> "UncertainArray":
        """Reconstruct an UncertainArray from shared memory (zero-copy read).

        Copies the data out of the shared segments before returning so the
        caller does not need to keep them alive.

        Parameters
        ----------
        values_shm_name, epsilons_shm_name : str
            Segment names returned by :meth:`to_shared_memory`.
        meta_json : str
            JSON metadata string returned by :meth:`to_shared_memory`.
        """
        if not _HAS_NUMPY:
            raise ImportError("numpy is required for shared memory support")

        meta = json.loads(meta_json)
        shape = tuple(meta["shape"])

        v_shm = SharedMemory(name=values_shm_name, create=False)
        e_shm = SharedMemory(name=epsilons_shm_name, create=False)

        # .copy() so we own the data after closing the segments.
        values = np.ndarray(shape, dtype=np.float64, buffer=v_shm.buf).copy()
        epsilons = np.ndarray(shape, dtype=np.float64, buffer=e_shm.buf).copy()

        v_shm.close()
        e_shm.close()

        return cls(values, epsilons, meta.get("provenance", ""))

    @staticmethod
    def free_shared_memory(values_shm_name: str, epsilons_shm_name: str) -> None:
        """Unlink shared memory segments created by :meth:`to_shared_memory`.

        Safe to call even if the segments have already been unlinked.
        """
        for name in (values_shm_name, epsilons_shm_name):
            try:
                shm = SharedMemory(name=name, create=False)
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass

    # ---- Representation ---------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"UncertainArray(n={len(self)}, "
            f"mean={np.mean(self.values):.4g}, "
            f"mean_eps={np.mean(self.epsilons):.4g}, "
            f"prov={self.provenance!r})"
        )
