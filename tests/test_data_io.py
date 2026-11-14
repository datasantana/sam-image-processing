import hashlib
import zipfile

import pytest

from segpipe import data_io
from segpipe.data_io import DatasetError


def test_verify_archive_ok_and_failures(pack, tmp_path):
    zpath, cfg = pack
    a = cfg["archive"]
    data_io.verify_archive(zpath, a["size_bytes"], a["sha256"], show_progress=False)
    with pytest.raises(DatasetError, match="Tamaño"):
        data_io.verify_archive(zpath, a["size_bytes"] + 1, a["sha256"], show_progress=False)
    with pytest.raises(DatasetError, match="sha256"):
        data_io.verify_archive(zpath, a["size_bytes"], "0" * 64, show_progress=False)


def test_ensure_dataset_from_env_zip_then_cached(pack, tmp_path, monkeypatch):
    zpath, cfg = pack
    monkeypatch.setenv("SEGPACK_ZIP", str(zpath))
    root = tmp_path / "vm"
    monkeypatch.setenv("SEGPACK_ROOT", str(root))
    d = data_io.ensure_dataset(cfg, verbose=False)
    assert d == root / "segpack_test"
    assert (d / data_io.MARKER).is_file()
    # segunda vez: paso 1 (ya validado), aunque el zip de origen desaparezca
    monkeypatch.delenv("SEGPACK_ZIP")
    zpath.unlink()
    assert data_io.ensure_dataset(cfg, verbose=False) == d


def test_ensure_dataset_rejects_corrupt_zip(pack, tmp_path, monkeypatch):
    zpath, cfg = pack
    cfg["archive"]["sha256"] = "f" * 64
    monkeypatch.setenv("SEGPACK_ZIP", str(zpath))
    monkeypatch.setenv("SEGPACK_ROOT", str(tmp_path / "vm"))
    with pytest.raises(DatasetError, match="sha256"):
        data_io.ensure_dataset(cfg, verbose=False)


def test_ensure_dataset_without_sources_fails_clearly(pack, tmp_path, monkeypatch):
    _, cfg = pack
    monkeypatch.setenv("SEGPACK_ROOT", str(tmp_path / "empty"))
    with pytest.raises(DatasetError, match="SEGPACK_ZIP"):
        data_io.ensure_dataset(cfg, verbose=False)


def test_validation_detects_missing_mask_and_size(pack, tmp_path):
    zpath, cfg = pack
    d = data_io.safe_extract(zpath, tmp_path / "x")
    assert data_io.validate_extracted(d, cfg).ok
    masks = sorted((d / "masks").iterdir())
    masks[0].unlink()
    from PIL import Image
    Image.new("RGB", (10, 10)).save(masks[1])
    rep = data_io.validate_extracted(d, cfg)
    assert not rep.ok
    txt = "\n".join(rep.problems)
    assert "falta máscara" in txt and "máscara (10, 10)" in txt


def test_env_dir_with_tampered_image_fails(pack, tmp_path, monkeypatch):
    zpath, cfg = pack
    d = data_io.safe_extract(zpath, tmp_path / "x")
    img = sorted((d / "images").iterdir())[0]
    raw = bytearray(img.read_bytes()); raw[-3] ^= 0xFF; img.write_bytes(bytes(raw))
    monkeypatch.setenv("SEGPACK_DIR", str(d))
    monkeypatch.setenv("SEGPACK_ROOT", str(tmp_path / "vm"))
    with pytest.raises(DatasetError, match="sha256 de imagen"):
        data_io.ensure_dataset(cfg, verbose=False)


def test_safe_extract_rejects_zip_slip(tmp_path):
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("../evil.txt", "x")
    with pytest.raises(DatasetError, match="insegura"):
        data_io.safe_extract(z, tmp_path / "out")


def test_download_ranges_resumes_after_interruption(tmp_path, monkeypatch):
    monkeypatch.setattr(data_io.time, "sleep", lambda s: None)
    payload = bytes(range(256)) * 40  # 10240 bytes
    calls = []
    state = {"fail_at": 4096}

    def fetch(start, end):
        calls.append(start)
        if state["fail_at"] is not None and start >= state["fail_at"]:
            raise ConnectionError("sesión cortada")
        return payload[start:end + 1]

    out = tmp_path / "f.zip"
    with pytest.raises(DatasetError):
        data_io.download_ranges(fetch, len(payload), out, chunk=1024, retries=1)
    part = tmp_path / "f.zip.part"
    assert part.stat().st_size == 4096 and not out.exists()
    state["fail_at"] = None
    calls.clear()
    data_io.download_ranges(fetch, len(payload), out, chunk=1024)
    assert calls[0] == 4096  # reanuda, no empieza de cero
    assert out.read_bytes() == payload and not part.exists()


def test_download_from_drive_uses_get_media_with_ranges(tmp_path):
    payload = b"abcdefghij" * 1000

    class Req:
        def __init__(self, fn): self.fn, self.headers = fn, {}
        def execute(self): return self.fn(self.headers)

    class Files:
        def get(self, fileId, fields, supportsAllDrives):
            assert fileId == "ID"
            return Req(lambda h: {"name": "x.zip", "size": str(len(payload))})
        def get_media(self, fileId, supportsAllDrives):
            def run(h):
                a, b = h["Range"].split("=")[1].split("-")
                return payload[int(a):int(b) + 1]
            return Req(run)

    class Svc:
        def files(self): return Files()

    out = data_io.download_from_drive("ID", tmp_path / "x.zip", expected_size=len(payload), service=Svc())
    assert hashlib.sha256(out.read_bytes()).hexdigest() == hashlib.sha256(payload).hexdigest()
    with pytest.raises(DatasetError, match="mide"):
        data_io.download_from_drive("ID", tmp_path / "y.zip", expected_size=5, service=Svc())
