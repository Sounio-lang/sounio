"""Host-side checks for the Arrow GPU feed.

A CUDA device is not required. The checks lock the copy shape: a matrix
batch aliases the mapped Arrow buffer, a logical batch that crosses two
IPC record batches still has the right values, and the DataLoader does
not stack those batches a second time.
"""

import os
import sys
import tempfile
import unittest

_TOOLS = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

import arrow_feed.loader as loader
from arrow_feed.loader import (
    ArrowBatchDataset,
    DevicePrefetcher,
    _shard,
    _spans,
    make_dataloader,
    payload_bytes,
    rows_for_payload,
    write_feature_file,
)


def _collect(dataset):
    """Run ``dataset`` and keep each tensor next to the array it was built from.

    ``Tensor.numpy()`` is a view of the tensor, so its ``OWNDATA`` flag is
    always false and says nothing about the Arrow buffer. The array handed
    to ``from_numpy`` is the one whose flags and address matter.
    """
    captured = []
    original = loader._tensor_from_numpy

    def wrap(array):
        tensor = original(array)
        captured.append((array, tensor))
        return tensor

    loader._tensor_from_numpy = wrap
    try:
        batches = list(dataset)
    finally:
        loader._tensor_from_numpy = original
    return batches, captured


def _write_two_batches(path, first, second, labels_first, labels_second):
    def batch(matrix, labels):
        packed = np.ascontiguousarray(matrix, dtype=np.float32)
        values = pa.array(packed.reshape(-1), type=pa.float32())
        lists = pa.FixedSizeListArray.from_arrays(values, packed.shape[1])
        lab = pa.array(np.ascontiguousarray(labels, dtype=np.int64), type=pa.int64())
        return pa.record_batch([lists, lab], names=["features", "label"])

    schema = batch(first, labels_first).schema
    with pa.OSFile(path, "wb") as sink:
        writer = ipc.new_file(sink, schema)
        writer.write_batch(batch(first, labels_first))
        writer.write_batch(batch(second, labels_second))
        writer.close()


class ArrowFeedTest(unittest.TestCase):
    def test_matrix_batch_aliases_the_arrow_buffer(self):
        rows, width = 32, 16
        features = np.arange(rows * width, dtype=np.float64).reshape(rows, width)
        labels = np.arange(rows, dtype=np.int64) * 3
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "features.arrow")
            write_feature_file(path, features, labels)
            seen, captured = _collect(ArrowBatchDataset(path, batch_size=8))
        self.assertEqual(len(seen), 4)
        first = seen[0]
        self.assertEqual(tuple(first["features"].shape), (8, width))
        self.assertEqual(first["features"].dtype, __import__("torch").float32)
        self.assertEqual(tuple(first["label"].shape), (8,))
        np.testing.assert_allclose(
            first["features"].numpy(), features[:8].astype(np.float32)
        )
        np.testing.assert_array_equal(first["label"].numpy(), labels[:8])
        view, tensor = captured[0]
        self.assertIs(tensor, first["features"])
        self.assertFalse(view.flags["OWNDATA"])
        self.assertEqual(tensor.data_ptr(), view.ctypes.data)
        self.assertEqual(payload_bytes(first["features"]), 8 * width * 4)

    def test_partial_tail_and_drop_last(self):
        features = np.arange(10 * 2, dtype=np.float32).reshape(10, 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tail.arrow")
            write_feature_file(path, features)
            kept = list(ArrowBatchDataset(path, batch_size=4, drop_last=False))
            dropped = list(ArrowBatchDataset(path, batch_size=4, drop_last=True))
        self.assertEqual([tuple(b["features"].shape) for b in kept], [(4, 2), (4, 2), (2, 2)])
        self.assertEqual(len(dropped), 2)
        np.testing.assert_array_equal(kept[-1]["features"].numpy(), features[8:])

    def test_span_across_ipc_batches_keeps_values(self):
        first = np.arange(10 * 3, dtype=np.float32).reshape(10, 3)
        second = np.arange(10 * 3, dtype=np.float32).reshape(10, 3) + 100
        whole = np.concatenate([first, second], axis=0)
        labels = np.arange(20, dtype=np.int64)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "split.arrow")
            _write_two_batches(path, first, second, labels[:10], labels[10:])
            # batch_size 6 crosses the record-batch boundary at row 10.
            batches, captured = _collect(ArrowBatchDataset(path, batch_size=6))
        self.assertEqual(len(batches), 4)
        got = np.concatenate([b["features"].numpy() for b in batches], axis=0)
        got_y = np.concatenate([b["label"].numpy() for b in batches], axis=0)
        np.testing.assert_array_equal(got, whole)
        np.testing.assert_array_equal(got_y, labels)
        # Each batch records features, then label. The crossing batch owns
        # its storage. The first batch still aliases the mapped buffer.
        cross_view, cross_tensor = captured[2]
        first_view, first_tensor = captured[0]
        self.assertTrue(cross_view.flags["OWNDATA"])
        self.assertEqual(cross_tensor.data_ptr(), cross_view.ctypes.data)
        self.assertFalse(first_view.flags["OWNDATA"])
        self.assertEqual(first_tensor.data_ptr(), first_view.ctypes.data)

    def test_column_table_packs_one_matrix(self):
        n = 5
        a = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        b = np.array([10, 20, 30, 40, 50], dtype=np.float64)
        batch = pa.record_batch(
            [pa.array(a), pa.array(b), pa.array(np.arange(n, dtype=np.int64))],
            names=["a", "b", "label"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cols.arrow")
            with pa.OSFile(path, "wb") as sink:
                writer = ipc.new_file(sink, batch.schema)
                writer.write_batch(batch)
                writer.close()
            got, captured = _collect(ArrowBatchDataset(path, batch_size=5))
        self.assertEqual(len(got), 1)
        expect = np.column_stack([a, b.astype(np.float32)])
        np.testing.assert_array_equal(got[0]["features"].numpy(), expect)
        packed, tensor = captured[0]
        self.assertTrue(packed.flags["OWNDATA"])
        self.assertEqual(tensor.data_ptr(), packed.ctypes.data)

    def test_nulls_are_refused(self):
        values = pa.array([1.0, 2.0, None, 4.0], type=pa.float32())
        batch = pa.record_batch([values], names=["x"])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nulls.arrow")
            with pa.OSFile(path, "wb") as sink:
                writer = ipc.new_file(sink, batch.schema)
                writer.write_batch(batch)
                writer.close()
            with self.assertRaises(ValueError):
                list(ArrowBatchDataset(path, batch_size=2))

    def test_dataloader_does_not_restack(self):
        features = np.arange(16 * 4, dtype=np.float32).reshape(16, 4)
        labels = np.arange(16, dtype=np.int64)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "loader.arrow")
            write_feature_file(path, features, labels)
            torch_loader = make_dataloader(
                path, batch_size=8, num_workers=0, pin_memory=False
            )
            batches, captured = _collect(torch_loader)
        self.assertEqual(len(batches), 2)
        self.assertEqual(tuple(batches[0]["features"].shape), (8, 4))
        self.assertNotIn("aliases_arrow", batches[0])
        np.testing.assert_array_equal(batches[1]["label"].numpy(), labels[8:])
        # batch_size=None must hand the Arrow tensor through, not restack it.
        self.assertEqual(batches[0]["features"].data_ptr(), captured[0][0].ctypes.data)

    def test_prefetcher_on_cpu_returns_the_batches(self):
        features = np.ones((6, 2), dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cpu.arrow")
            write_feature_file(path, features)
            prefetched = list(
                DevicePrefetcher(ArrowBatchDataset(path, batch_size=4), device="cpu")
            )
        self.assertEqual(len(prefetched), 2)
        np.testing.assert_array_equal(prefetched[0]["features"].numpy(), features[:4])

    def test_shuffle_batches_is_a_permutation(self):
        features = np.arange(20 * 1, dtype=np.float32).reshape(20, 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "shuf.arrow")
            write_feature_file(path, features)
            plain = list(ArrowBatchDataset(path, batch_size=4))
            shuffled = list(
                ArrowBatchDataset(path, batch_size=4, shuffle_batches=True, seed=7)
            )
        plain_rows = [int(b["features"][0, 0].item()) for b in plain]
        shuf_rows = [int(b["features"][0, 0].item()) for b in shuffled]
        self.assertEqual(sorted(plain_rows), sorted(shuf_rows))
        self.assertNotEqual(plain_rows, shuf_rows)

    def test_workers_cover_each_batch_once(self):
        features = np.arange(20, dtype=np.float32).reshape(20, 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "workers.arrow")
            write_feature_file(path, features)
            torch_loader = make_dataloader(
                path, batch_size=4, num_workers=2, pin_memory=False
            )
            try:
                batches = list(torch_loader)
            finally:
                del torch_loader
        starts = sorted(int(batch["features"][0, 0].item()) for batch in batches)
        self.assertEqual(starts, [0, 4, 8, 12, 16])
        self.assertEqual(tuple(batches[0]["features"].shape), (4, 1))

    def test_shard_covers_every_span_once(self):
        spans = _spans(100, 10, drop_last=False)
        self.assertEqual(len(spans), 10)
        covered = []
        for worker in range(3):
            covered.extend(_shard(spans, worker, 3))
        self.assertEqual(covered, spans)

    def test_rows_for_payload_is_the_byte_floor(self):
        # 8 MiB / (64 * 4) = 32768 rows exactly.
        self.assertEqual(rows_for_payload(64, target_bytes=8 * 1024 * 1024), 32768)
        self.assertEqual(rows_for_payload(3, target_bytes=10), 1)

    def test_float64_matrix_is_refused_on_read(self):
        matrix = np.zeros((4, 2), dtype=np.float64)
        values = pa.array(matrix.reshape(-1), type=pa.float64())
        lists = pa.FixedSizeListArray.from_arrays(values, 2)
        batch = pa.record_batch([lists], names=["features"])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "wide.arrow")
            with pa.OSFile(path, "wb") as sink:
                writer = ipc.new_file(sink, batch.schema)
                writer.write_batch(batch)
                writer.close()
            with self.assertRaises(ValueError):
                list(ArrowBatchDataset(path, batch_size=2))


if __name__ == "__main__":
    unittest.main()
