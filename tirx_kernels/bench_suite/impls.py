# Copyright (c) 2026 The TIRx Authors
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

"""Implementation-name classification shared by bench-suite reports."""

from collections.abc import Mapping


def is_our_impl(name: str) -> bool:
    """Whether ``name`` identifies a TIR/TIRx implementation."""
    return name in {"tir", "tirx"} or name.startswith(("tir_", "tirx_"))


def our_impls(impls: Mapping[str, float]) -> list[str]:
    """Return all TIR/TIRx implementation names in result order."""
    return [name for name in impls if is_our_impl(name)]
