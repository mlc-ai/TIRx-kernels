# Copyright (c) 2026 The TIRX Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from types import SimpleNamespace

from tirx_kernels.flashinfer.utils import _flashkda_bench


def test_raw_peer_accepts_installed_package_version(monkeypatch, tmp_path):
    package_path = tmp_path / "__init__.py"
    extension_path = tmp_path / "flash_kda_C.so"
    package_path.write_text("# package\n")
    extension_path.write_bytes(b"extension")

    class Distribution:
        version = "9.8.7+arbitrary"

        @staticmethod
        def read_text(_name):
            return None

    modules = {
        "flash_kda": SimpleNamespace(__file__=str(package_path)),
        "flash_kda_C": SimpleNamespace(__file__=str(extension_path)),
    }
    monkeypatch.delenv("FLASHKDA_SOURCE_DIR", raising=False)
    monkeypatch.setattr(_flashkda_bench, "distribution", lambda _name: Distribution())
    monkeypatch.setattr(_flashkda_bench, "import_module", modules.__getitem__)

    package, provenance = _flashkda_bench._load_flash_kda_peer()

    assert package is modules["flash_kda"]
    assert provenance["package_version"] == Distribution.version
    assert "source_commit" not in provenance
    assert "cutlass_commit" not in provenance
