from pathlib import Path

from nebulalm.packing.packer import pack_documents, read_shard, write_shard


def test_window_edges_and_pad_mask(tmp_path: Path):
    eos, pad = 1, 0
    cases = {
        254: 1,
        255: 0,
        256: 255,
    }
    for content_len, expected_pad in cases.items():
        windows = pack_documents([("doc", list(range(2, 2 + content_len)), content_len)], eos_id=eos, pad_id=pad)
        if content_len + 1 <= 256:
            assert len(windows) == 1
            window = windows[0]
            assert window["input_ids"].count(pad) == expected_pad
            assert sum(1 for token, mask in zip(window["input_ids"], window["loss_mask"]) if token == pad and mask) == 0
            assert window["input_ids"].count(eos) == 1
            assert window["byte_count"] == content_len
        else:
            assert len(windows) == 2
            assert sum(window["byte_count"] for window in windows) == content_len
            assert sum(window["input_ids"].count(eos) for window in windows) == 1
    windows = pack_documents([("doc-a", [2, 3, 4], 3), ("doc-b", [5, 6], 2)], eos_id=eos, pad_id=pad)
    assert windows[0]["segment_ids"][0] != windows[0]["segment_ids"][4]
    destination = tmp_path / "train.bin"
    write_shard(destination, windows)
    loaded = read_shard(destination)
    assert loaded[0]["input_ids"] == windows[0]["input_ids"]
    assert loaded[0]["loss_mask"] == windows[0]["loss_mask"]
