# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import os
import pickle
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import pytest
from marshmallow import Schema

from superset.dashboards.permalink.schemas import DashboardPermalinkSchema
from superset.key_value.exceptions import (
    KeyValueCodecDecodeException,
    KeyValueCodecEncodeException,
)
from superset.key_value.types import (
    BinaryKeyValueCodec,
    JsonKeyValueCodec,
    MarshmallowKeyValueCodec,
    PickleKeyValueCodec,
)


@pytest.mark.parametrize(
    "input_,expected_result",
    [
        (
            {"foo": "bar"},
            {"foo": "bar"},
        ),
        (
            {"foo": (1, 2, 3)},
            {"foo": [1, 2, 3]},
        ),
        (
            {1, 2, 3},
            KeyValueCodecEncodeException(),
        ),
        (
            object(),
            KeyValueCodecEncodeException(),
        ),
    ],
)
def test_json_codec(input_: Any, expected_result: Any):
    cm = (
        pytest.raises(type(expected_result))
        if isinstance(expected_result, Exception)
        else nullcontext()
    )
    with cm:
        codec = JsonKeyValueCodec()
        encoded_value = codec.encode(input_)
        assert expected_result == codec.decode(encoded_value)


@pytest.mark.parametrize(
    "schema,input_,expected_result",
    [
        (
            DashboardPermalinkSchema(),
            {
                "dashboardId": "1",
                "state": {
                    "urlParams": [["foo", "bar"], ["foo", "baz"]],
                },
            },
            {
                "dashboardId": "1",
                "state": {
                    "urlParams": [("foo", "bar"), ("foo", "baz")],
                },
            },
        ),
        (
            DashboardPermalinkSchema(),
            {"foo": "bar"},
            KeyValueCodecEncodeException(),
        ),
    ],
)
def test_marshmallow_codec(schema: Schema, input_: Any, expected_result: Any):
    cm = (
        pytest.raises(type(expected_result))
        if isinstance(expected_result, Exception)
        else nullcontext()
    )
    with cm:
        codec = MarshmallowKeyValueCodec(schema)
        encoded_value = codec.encode(input_)
        assert expected_result == codec.decode(encoded_value)


@pytest.mark.parametrize(
    "input_,expected_result",
    [
        (
            {1, 2, 3},
            {1, 2, 3},
        ),
        (
            {"foo": 1, "bar": {1: (1, 2, 3)}, "baz": {1, 2, 3}},
            {
                "foo": 1,
                "bar": {1: (1, 2, 3)},
                "baz": {1, 2, 3},
            },
        ),
        (
            {
                "uuid": UUID("7b4a1b1a-1c1a-4a0a-9c3a-1b1a1c1a4a0a"),
                "when": datetime(2023, 5, 1, 12, 3, tzinfo=timezone.utc),
                "complex": complex(1, 1),
                "frozen": frozenset({1, 2}),
            },
            {
                "uuid": UUID("7b4a1b1a-1c1a-4a0a-9c3a-1b1a1c1a4a0a"),
                "when": datetime(2023, 5, 1, 12, 3, tzinfo=timezone.utc),
                "complex": complex(1, 1),
                "frozen": frozenset({1, 2}),
            },
        ),
    ],
)
def test_pickle_codec(input_: Any, expected_result: Any):
    codec = PickleKeyValueCodec()
    encoded_value = codec.encode(input_)
    assert expected_result == codec.decode(encoded_value)


class Exploit:
    """Payload whose reconstruction would shell out to `os.system`."""

    def __init__(self, command: str):
        self.command = command

    def __reduce__(self):
        return os.system, (self.command,)


@pytest.mark.parametrize(
    "module,name",
    [
        ("os", "system"),
        ("subprocess", "check_output"),
        ("builtins", "eval"),
    ],
)
def test_pickle_codec_rejects_disallowed_globals(module: str, name: str):
    # protocol 0 GLOBAL opcode, i.e. a bare reference to `module.name`
    payload = f"c{module}\n{name}\n.".encode()
    with pytest.raises(KeyValueCodecDecodeException):
        PickleKeyValueCodec().decode(payload)


def test_pickle_codec_rejects_command_execution_payload(tmp_path):
    marker = tmp_path / "pwned"
    payload = pickle.dumps(Exploit(f"touch {marker}"))
    with pytest.raises(KeyValueCodecDecodeException):
        PickleKeyValueCodec().decode(payload)
    assert not marker.exists()


def test_binary_codec_encode():
    codec = BinaryKeyValueCodec()
    raw = b"\x00\x01binary\xffdata"
    assert codec.encode(raw) == raw


def test_binary_codec_round_trips():
    codec = BinaryKeyValueCodec()
    raw = b"\x00\x01binary\xffdata"
    assert codec.decode(raw) == raw
    assert codec.encode(codec.decode(raw)) == raw
