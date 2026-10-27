"""Arrow IPC batches for a torch DataLoader.

The native Sounio IPC codec is not this path. It refuses a batch past 16
rows or a 4096-byte body. A GPU copy that small does not fill the bus.
"""

from arrow_feed.loader import (
    ArrowBatchDataset,
    DevicePrefetcher,
    make_dataloader,
    payload_bytes,
    rows_for_payload,
    write_feature_file,
)

__all__ = [
    "ArrowBatchDataset",
    "DevicePrefetcher",
    "make_dataloader",
    "payload_bytes",
    "rows_for_payload",
    "write_feature_file",
]
