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

import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.exc import IntegrityError

from superset.security.session_invalidation import (
    _as_utc_timestamp,
    enforce_session_validity,
    invalidate_user_sessions,
    is_session_invalidated,
    SESSION_LOGIN_AT_KEY,
)


def test_no_epoch_is_never_invalidated() -> None:
    """A user that was never disabled (NULL epoch) is never invalidated."""
    assert is_session_invalidated(login_at=None, invalidated_at=None) is False
    assert is_session_invalidated(login_at=1_000.0, invalidated_at=None) is False


def test_epoch_with_no_login_time_fails_closed() -> None:
    """A pre-feature session (no _login_at) on a disabled user is invalidated."""
    epoch = datetime.now(timezone.utc)
    assert is_session_invalidated(login_at=None, invalidated_at=epoch) is True


def test_session_before_epoch_is_invalidated() -> None:
    epoch = datetime.now(timezone.utc)
    before = (epoch - timedelta(minutes=5)).timestamp()
    assert is_session_invalidated(login_at=before, invalidated_at=epoch) is True


def test_session_after_epoch_is_valid() -> None:
    """A fresh login after a disable+re-enable must not be invalidated."""
    epoch = datetime.now(timezone.utc)
    after = (epoch + timedelta(minutes=5)).timestamp()
    assert is_session_invalidated(login_at=after, invalidated_at=epoch) is False


def test_login_exactly_at_epoch_is_valid() -> None:
    epoch = datetime.now(timezone.utc)
    assert (
        is_session_invalidated(login_at=epoch.timestamp(), invalidated_at=epoch)
        is False
    )


def test_naive_epoch_is_treated_as_utc() -> None:
    """
    The DB column is a naive UTC ``DateTime``; the comparison must treat it as
    UTC, not local time (otherwise it skews by the local offset).
    """
    aware = datetime(2026, 6, 2, 12, 0, 0, tzinfo=timezone.utc)
    naive = aware.replace(tzinfo=None)
    assert _as_utc_timestamp(naive) == aware.timestamp()

    just_before = aware.timestamp() - 1
    just_after = aware.timestamp() + 1
    assert is_session_invalidated(login_at=just_before, invalidated_at=naive) is True
    assert is_session_invalidated(login_at=just_after, invalidated_at=naive) is False


@pytest.fixture
def process_tz(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    """Set the process time zone via ``TZ`` for a test, restoring it after.

    ``time.tzset`` mutates C-library state that ``monkeypatch`` cannot undo on
    its own, so re-apply it once the env var is restored to avoid leaking the
    zone into later tests.
    """

    def _set(tz: str) -> None:
        monkeypatch.setenv("TZ", tz)
        time.tzset()

    yield _set
    monkeypatch.undo()
    time.tzset()


# Zones with non-zero UTC offsets, including one east and one west of UTC and
# one with a fractional-hour offset, so a naive value mistakenly read as local
# time would visibly skew the epoch in either direction.
@pytest.mark.parametrize(
    "tz",
    ["America/Los_Angeles", "Asia/Kolkata", "Europe/Berlin", "Pacific/Chatham"],
)
def test_naive_utc_matches_aware_utc_under_non_utc_tz(tz: str, process_tz) -> None:  # noqa: ANN001
    """
    The naive/aware boundary must not move with the process time zone.

    ``sessions_invalidated_at`` is stored as a naive UTC ``DateTime``. Calling
    ``.timestamp()`` on a naive datetime interprets it in the local zone, so if
    ``_as_utc_timestamp`` did not force UTC the epoch would skew by the local
    UTC offset under a non-UTC ``TZ`` — invalidating sessions early or never.
    Pin that a naive-UTC input yields the same epoch as the equivalent aware-UTC
    input, and that the ``is_session_invalidated`` boundary stays put, while
    ``TZ`` is set to a non-UTC zone.
    """
    process_tz(tz)

    # Sanity-check that the chosen zone really is offset from UTC in this
    # process, so the assertions below are actually exercising the skew guard.
    assert time.timezone != 0 or time.altzone != 0

    aware = datetime(2026, 6, 2, 12, 0, 0, tzinfo=timezone.utc)
    naive = aware.replace(tzinfo=None)

    assert _as_utc_timestamp(naive) == aware.timestamp()
    assert _as_utc_timestamp(naive) == _as_utc_timestamp(aware)

    just_before = aware.timestamp() - 1
    just_after = aware.timestamp() + 1
    assert is_session_invalidated(login_at=just_before, invalidated_at=naive) is True
    assert is_session_invalidated(login_at=just_after, invalidated_at=naive) is False
    # The naive column and its aware equivalent invalidate identically.
    assert is_session_invalidated(
        login_at=just_before, invalidated_at=naive
    ) is is_session_invalidated(login_at=just_before, invalidated_at=aware)


MODULE = "superset.security.session_invalidation"


def _user(*, authenticated: bool = True, guest: bool = False) -> SimpleNamespace:
    return SimpleNamespace(is_authenticated=authenticated, is_guest_user=guest)


def test_enforce_skips_unauthenticated_user() -> None:
    """No authenticated user ⇒ nothing to enforce, request proceeds."""
    with (
        patch(f"{MODULE}.current_user", _user(authenticated=False)),
        patch(f"{MODULE}.logout_user") as logout,
    ):
        assert enforce_session_validity() is None
        logout.assert_not_called()


def test_enforce_skips_guest_user() -> None:
    """Guest (embedded) users have their own revocation path and are skipped."""
    with (
        patch(f"{MODULE}.current_user", _user(guest=True)),
        patch(f"{MODULE}._get_user_invalidated_at") as get_epoch,
        patch(f"{MODULE}.logout_user") as logout,
    ):
        assert enforce_session_validity() is None
        get_epoch.assert_not_called()
        logout.assert_not_called()


def test_enforce_no_epoch_leaves_session_alone() -> None:
    """A user with no invalidation epoch is never logged out."""
    with (
        patch(f"{MODULE}.current_user", _user()),
        patch(f"{MODULE}._get_user_invalidated_at", return_value=None),
        patch(f"{MODULE}.logout_user") as logout,
    ):
        assert enforce_session_validity() is None
        logout.assert_not_called()


def test_enforce_valid_session_is_not_logged_out() -> None:
    """A session that logged in after the epoch stays authenticated."""
    epoch = datetime.now(timezone.utc)
    after = (epoch + timedelta(minutes=5)).timestamp()
    fake_session = MagicMock()
    fake_session.get.return_value = after
    with (
        patch(f"{MODULE}.current_user", _user()),
        patch(f"{MODULE}._get_user_invalidated_at", return_value=epoch),
        patch(f"{MODULE}.session", fake_session),
        patch(f"{MODULE}.logout_user") as logout,
    ):
        assert enforce_session_validity() is None
        fake_session.get.assert_called_once_with(SESSION_LOGIN_AT_KEY)
        logout.assert_not_called()


def test_enforce_invalidated_session_is_logged_out() -> None:
    """A session predating the epoch is cleared and flashed a warning."""
    epoch = datetime.now(timezone.utc)
    before = (epoch - timedelta(minutes=5)).timestamp()
    fake_session = MagicMock()
    fake_session.get.return_value = before
    with (
        patch(f"{MODULE}.current_user", _user()),
        patch(f"{MODULE}._get_user_invalidated_at", return_value=epoch),
        patch(f"{MODULE}.session", fake_session),
        patch(f"{MODULE}.logout_user") as logout,
        patch(f"{MODULE}.flash") as flash,
    ):
        assert enforce_session_validity() is None
        logout.assert_called_once()
        fake_session.clear.assert_called_once()
        flash.assert_called_once()


def test_enforce_fails_open_on_error() -> None:
    """Any error in the check logs a warning and allows the request."""
    # A real (non-guest, authenticated) user so the check reaches the epoch
    # lookup, which then raises — exercising the fail-open handler.
    user = SimpleNamespace(is_authenticated=True, is_guest_user=False)
    with (
        patch(f"{MODULE}.current_user", user),
        patch(f"{MODULE}._get_user_invalidated_at", side_effect=RuntimeError("boom")),
        patch(f"{MODULE}.logout_user") as logout,
        patch(f"{MODULE}.logger") as logger,
    ):
        assert enforce_session_validity() is None
        logout.assert_not_called()
        logger.warning.assert_called_once()


def test_invalidate_updates_existing_row() -> None:
    """When a row already exists, the upsert updates it and skips the insert."""
    connection = MagicMock()
    connection.execute.return_value.rowcount = 1
    invalidate_user_sessions(connection, user_id=7)
    # Single UPDATE, no INSERT / SAVEPOINT.
    assert connection.execute.call_count == 1
    connection.begin_nested.assert_not_called()


def test_invalidate_inserts_when_missing() -> None:
    """When no row exists, the upsert inserts one inside a SAVEPOINT."""
    connection = MagicMock()
    # First execute is the UPDATE (rowcount 0); second is the INSERT.
    connection.execute.return_value.rowcount = 0
    invalidate_user_sessions(connection, user_id=7)
    assert connection.execute.call_count == 2
    connection.begin_nested.assert_called_once()


def test_invalidate_retries_as_update_on_race() -> None:
    """If a concurrent disable wins the insert, the IntegrityError is caught
    and the row is stamped via UPDATE instead of duplicating it."""
    connection = MagicMock()
    update_result = SimpleNamespace(rowcount=0)
    retry_result = SimpleNamespace(rowcount=1)

    calls: list[str] = []

    def execute(statement, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        compiled = str(statement).strip().upper()
        if compiled.startswith("UPDATE"):
            calls.append("update")
            # First UPDATE finds nothing; the retry UPDATE succeeds.
            return retry_result if len(calls) > 1 else update_result
        calls.append("insert")
        raise IntegrityError("insert", {}, Exception())

    connection.execute.side_effect = execute
    invalidate_user_sessions(connection, user_id=7)
    # update (miss) -> insert (race loses) -> update (retry succeeds)
    assert calls == ["update", "insert", "update"]
