"""Feed a torch training step from an Arrow IPC file.

Host-to-device bandwidth wants one large DMA, not a row walk. This module
memory-maps an uncompressed Arrow IPC file and slices each step as one
contiguous float32 matrix of shape ``[batch, features]``.

Copy chain when CUDA is present:

1. The batch is a zero-copy view of the mapped Arrow values buffer
   (FixedSizeList<float32>, one row-major matrix). A table of separate
   float columns is packed once into that same shape.
2. ``DataLoader(pin_memory=True)`` copies that view into page-locked host
   memory on the pin thread, overlapped with the training step.
3. ``DevicePrefetcher`` copies the next batch to the GPU on a side stream
   with ``non_blocking=True`` while the current batch is in use.

The native Sounio codec in ``stdlib/arrow`` is a different surface. It
fails closed past 16 rows and a 4096-byte body, and it is not the C Data
Interface. Do not point this loader at those buffers.

Shuffle, when requested, reorders whole batches. Shuffling rows would
turn the DMA into a gather.
"""

from __future__ import annotations

import random
import warnings
from typing import Iterator, Optional

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc


_PAYLOAD_FLOOR = 8 * 1024 * 1024


def rows_for_payload(n_features: int, target_bytes: int = _PAYLOAD_FLOOR) -> int:
    """Row count whose float32 matrix is at least ``target_bytes``.

    A transfer of a few megabytes is large enough to occupy the bus. This
    is that size divided by the row width. It is not a measured bandwidth.
    """
    if n_features < 1:
        raise ValueError("n_features must be positive")
    if target_bytes < 1:
        raise ValueError("target_bytes must be positive")
    row_bytes = n_features * 4
    return max(1, (target_bytes + row_bytes - 1) // row_bytes)


def payload_bytes(features: "torch.Tensor") -> int:
    """Bytes in the feature matrix that a device copy would move."""
    return int(features.numel()) * int(features.element_size())


def write_feature_file(path: str, features, labels=None) -> None:
    """Write ``features`` of shape ``[rows, features]`` as an IPC file.

    Values are cast to float32 once, here. The training loop then slices
    the file and does not cast. ``labels``, when given, are one int64 per
    row and are stored beside the matrix.
    """
    matrix = np.asarray(features)
    if matrix.ndim != 2:
        raise ValueError("features must have shape [rows, features]")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("features must contain at least one row and one column")
    packed = np.ascontiguousarray(matrix, dtype=np.float32)
    values = pa.array(packed.reshape(-1), type=pa.float32())
    lists = pa.FixedSizeListArray.from_arrays(values, int(packed.shape[1]))
    arrays = [lists]
    names = ["features"]
    if labels is not None:
        lab = np.asarray(labels)
        if lab.shape != (packed.shape[0],):
            raise ValueError("labels must have one value per feature row")
        lab = np.ascontiguousarray(lab, dtype=np.int64)
        arrays.append(pa.array(lab, type=pa.int64()))
        names.append("label")
    batch = pa.record_batch(arrays, names=names)
    with pa.OSFile(path, "wb") as sink:
        writer = ipc.new_file(sink, batch.schema)
        writer.write_batch(batch)
        writer.close()


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "arrow_feed needs torch. Install it before constructing a DataLoader."
        ) from exc
    return torch


def _tensor_from_numpy(array: np.ndarray):
    torch = _torch()
    array = np.ascontiguousarray(array) if not array.flags["C_CONTIGUOUS"] else array
    if array.flags.writeable:
        return torch.from_numpy(array)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The given NumPy array is not writable",
        )
        return torch.from_numpy(array)


def _spans(n_rows: int, batch_size: int, drop_last: bool) -> list:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    spans = []
    pos = 0
    while pos + batch_size <= n_rows:
        spans.append((pos, batch_size))
        pos += batch_size
    if (not drop_last) and pos < n_rows:
        spans.append((pos, n_rows - pos))
    return spans


def _shard(spans: list, worker: int, n_workers: int) -> list:
    """Give each worker a contiguous run of batches so its reads stay sequential."""
    count = len(spans)
    if n_workers < 1:
        raise ValueError("n_workers must be positive")
    if worker < 0 or worker >= n_workers:
        raise ValueError("worker id is outside the worker count")
    base = count // n_workers
    extra = count % n_workers
    if worker < extra:
        start = worker * (base + 1)
        stop = start + base + 1
    else:
        start = extra * (base + 1) + (worker - extra) * base
        stop = start + base
    return spans[start:stop]


def _is_f32_matrix(data_type) -> bool:
    return (
        pa.types.is_fixed_size_list(data_type)
        and data_type.value_type == pa.float32()
        and data_type.list_size > 0
    )


def _is_float_primitive(data_type) -> bool:
    return (
        pa.types.is_float16(data_type)
        or pa.types.is_float32(data_type)
        or pa.types.is_float64(data_type)
    )


class _Index:
    """Row index over the record batches of one IPC file."""

    def __init__(self, batches: list):
        if not batches:
            self.batches = []
            self.starts = []
            self.n_rows = 0
            self.n_features = 0
            self.mode = "empty"
            self.has_label = False
            return
        schema = batches[0].schema
        for batch in batches[1:]:
            if batch.schema != schema:
                raise ValueError("record batches in one file must share a schema")
        names = list(schema.names)
        self.has_label = "label" in names
        if self.has_label and not pa.types.is_int64(schema.field("label").type):
            raise ValueError("label column must be int64")
        if "features" in names:
            extra = [name for name in names if name not in ("features", "label")]
            if extra:
                raise ValueError(
                    "unexpected columns %s; the matrix feed reads 'features' and an optional 'label'"
                    % extra
                )
            feat_type = schema.field("features").type
            if not _is_f32_matrix(feat_type):
                raise ValueError(
                    "features must be FixedSizeList<float32>; rewrite the file so the training loop does not cast"
                )
            self.mode = "matrix"
            self.n_features = int(feat_type.list_size)
            self.column_names = []
        else:
            self.column_names = [name for name in names if name != "label"]
            if not self.column_names:
                raise ValueError("file has no feature columns")
            for name in self.column_names:
                if not _is_float_primitive(schema.field(name).type):
                    raise ValueError(
                        "column %s is not float16, float32, or float64" % name
                    )
            self.mode = "columns"
            self.n_features = len(self.column_names)
        self.batches = batches
        self.starts = []
        total = 0
        for batch in batches:
            self.starts.append(total)
            total += batch.num_rows
        self.n_rows = total

    def _batch_of(self, row: int) -> int:
        lo = 0
        hi = len(self.starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.starts[mid] <= row:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _reject_nulls(self, column, name: str) -> None:
        if column.null_count != 0:
            raise ValueError(
                "column %s contains nulls; the GPU copy has no validity bitmap" % name
            )

    def _matrix_view(self, batch, local: int, count: int) -> np.ndarray:
        column = batch.column("features")
        self._reject_nulls(column, "features")
        values = column.values.to_numpy(zero_copy_only=True)
        begin = (column.offset + local) * self.n_features
        view = values[begin : begin + count * self.n_features]
        return view.reshape(count, self.n_features)

    def _primitive_view(self, batch, name: str, local: int, count: int) -> np.ndarray:
        column = batch.column(name)
        self._reject_nulls(column, name)
        sliced = column.slice(local, count)
        return sliced.to_numpy(zero_copy_only=True)

    def materialize(self, start: int, count: int):
        pieces = []
        labels = []
        remain = count
        pos = start
        while remain > 0:
            index = self._batch_of(pos)
            local = pos - self.starts[index]
            take = min(remain, self.batches[index].num_rows - local)
            batch = self.batches[index]
            if self.mode == "matrix":
                pieces.append(self._matrix_view(batch, local, take))
            else:
                packed = np.empty((take, self.n_features), dtype=np.float32)
                for col, name in enumerate(self.column_names):
                    packed[:, col] = self._primitive_view(batch, name, local, take)
                pieces.append(packed)
            if self.has_label:
                labels.append(self._primitive_view(batch, "label", local, take))
            pos += take
            remain -= take
        if len(pieces) == 1:
            features = pieces[0]
            label = labels[0] if labels else None
        else:
            features = np.concatenate(pieces, axis=0)
            label = np.concatenate(labels, axis=0) if labels else None
        return features, label


def _open_index(path: str) -> _Index:
    source = pa.memory_map(path, "r")
    reader = ipc.open_file(source)
    batches = [reader.get_batch(i) for i in range(reader.num_record_batches)]
    # The arrays own their buffers. Keep the map reachable from the index
    # so a worker cannot unmap a batch that a tensor still views.
    index = _Index(batches)
    index.source = source
    index.reader = reader
    return index


class ArrowBatchDataset(_torch().utils.data.IterableDataset):
    """Iterable of whole batches. Pair it with ``DataLoader(batch_size=None)``.

    Each yield is ``{"features": float32[B, F], "label": int64[B]}``.
    ``label`` is omitted when the file has no label column. A matrix file
    yields a tensor that aliases the mapped Arrow buffer. Do not write
    into that tensor. ``make_dataloader`` turns on ``pin_memory`` when CUDA
    is available, and that pin copy is the writable page-locked buffer.
    """

    def __init__(
        self,
        path: str,
        batch_size: int,
        drop_last: bool = False,
        shuffle_batches: bool = False,
        seed: Optional[int] = None,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.path = path
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle_batches = shuffle_batches
        self.seed = seed

    def __iter__(self) -> Iterator[dict]:
        torch = _torch()
        info = torch.utils.data.get_worker_info()
        worker = 0 if info is None else info.id
        n_workers = 1 if info is None else info.num_workers
        index = _open_index(self.path)
        spans = _spans(index.n_rows, self.batch_size, self.drop_last)
        spans = _shard(spans, worker, n_workers)
        if self.shuffle_batches and spans:
            seed = 0 if self.seed is None else self.seed
            random.Random(seed + worker).shuffle(spans)
        for start, count in spans:
            features, label = index.materialize(start, count)
            batch = {"features": _tensor_from_numpy(features)}
            if label is not None:
                batch["label"] = _tensor_from_numpy(label)
            yield batch


def make_dataloader(
    path: str,
    batch_size: int,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    drop_last: bool = False,
    shuffle_batches: bool = False,
    seed: Optional[int] = None,
    pin_memory: Optional[bool] = None,
):
    """DataLoader that yields Arrow batches already shaped for one device copy.

    ``batch_size`` is the Arrow row count, not a second collation. The
    DataLoader is constructed with ``batch_size=None`` so it does not stack
    those batches again. ``pin_memory`` defaults to whether CUDA is
    available. ``num_workers`` above zero splits the file into contiguous
    row ranges, one range per worker, and starts those workers with
    ``spawn`` so they do not fork a process that already owns threads.
    """
    torch = _torch()
    dataset = ArrowBatchDataset(
        path,
        batch_size,
        drop_last=drop_last,
        shuffle_batches=shuffle_batches,
        seed=seed,
    )
    if pin_memory is None:
        pin_memory = bool(torch.cuda.is_available())
    kwargs = {
        "batch_size": None,
        "pin_memory": pin_memory,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = True
        # fork after torch or pyarrow has started threads can deadlock.
        # spawn re-imports this module in each worker.
        kwargs["multiprocessing_context"] = "spawn"
    return torch.utils.data.DataLoader(dataset, **kwargs)


class DevicePrefetcher:
    """Overlap the next host-to-device copy with compute on the current batch.

    On CUDA the next batch is copied on a side stream. The consumer stream
    waits for that copy before the batch is returned, then the following
    batch starts moving. On CPU this is a plain iterator.
    """

    def __init__(self, batches, device):
        torch = _torch()
        self._torch = torch
        self._iter = iter(batches)
        if not isinstance(device, torch.device):
            device = torch.device(device)
        self.device = device
        self.stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        self._next = None
        self._preload()

    def _move(self, batch):
        if self.stream is None:
            return {
                key: value.to(self.device) for key, value in batch.items() if _is_tensor(value)
            }
        with self._torch.cuda.stream(self.stream):
            return {
                key: value.to(self.device, non_blocking=True)
                for key, value in batch.items()
                if _is_tensor(value)
            }

    def _preload(self) -> None:
        try:
            batch = next(self._iter)
        except StopIteration:
            self._next = None
            return
        self._next = self._move(batch)

    def __iter__(self):
        return self

    def __next__(self):
        if self._next is None:
            raise StopIteration
        if self.stream is not None:
            current = self._torch.cuda.current_stream(self.device)
            current.wait_stream(self.stream)
            for value in self._next.values():
                value.record_stream(current)
        batch = self._next
        self._preload()
        return batch


def _is_tensor(value) -> bool:
    torch = _torch()
    return isinstance(value, torch.Tensor)
