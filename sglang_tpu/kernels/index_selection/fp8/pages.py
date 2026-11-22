"""Packed FP8 index-key page geometry, shared by the kernels and KV pools."""

PAGE_RECORDS = 128
# A key's fields: 32 words of four FP8 features, then its FP32 scale.
KEY_ROWS = 32
PAGE_ROWS = KEY_ROWS + 1
# int32 rows a page occupies: the fields padded to whole 8-row tiles.
PAGE_STRIDE = 40
# Cache bytes per key: one int32 per row, padding included.
BYTES_PER_RECORD = PAGE_STRIDE * 4


def page_shape(pages: int) -> tuple[int, int, int]:
    return pages, PAGE_STRIDE, PAGE_RECORDS
