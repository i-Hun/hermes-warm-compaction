"""The loopback harness writes its synthetic session through the storage interface, with the older verb as fallback."""

from __future__ import annotations

import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from scripts import check_plugin_hermes as harness

ROW = {"role": "user", "content": "Synthetic question. Detail text."}
SUMMARY_ROW = {"role": "user", "content": "[CONTEXT SUMMARY]: Synthetic earlier summary.",
               "_compressed_summary": True, "display_kind": "compaction_notice"}

# The field names ``NewMessage`` takes, and the live key ``from_live`` maps onto them.
NEW_MESSAGE_FIELDS = frozenset({"role", "content", "timestamp", "tool_name", "tool_call_id", "display_kind",
                                "compressed_summary", "reasoning", "reasoning_content"})
LIVE_KEY_MAP = {"_compressed_summary": "compressed_summary"}


class NewMessage:
    """Stand-in for ``persistence.conversation.messages.NewMessage``.

    Its constructor refuses a field name the real dataclass does not declare, and ``from_live`` maps
    the live keys, so a helper that passes a live row straight to the constructor fails here as it
    would on the host.
    """

    def __init__(self, **fields):
        unknown = sorted(set(fields) - NEW_MESSAGE_FIELDS)
        if unknown:
            raise TypeError(f"NewMessage got an unknown field: {unknown[0]}")
        self.fields = fields

    @classmethod
    def from_live(cls, row):
        mapped = {LIVE_KEY_MAP.get(key, key): value for key, value in row.items()}
        return cls(**{key: value for key, value in mapped.items() if key in NEW_MESSAGE_FIELDS})


class NewSession:
    """Stand-in for ``persistence.conversation.sessions.NewSession``."""

    def __init__(self, **fields):
        self.fields = fields


class Store:
    """A store handle that records its writes and its close."""

    def __init__(self):
        self.sessions = SimpleNamespace(create=self.create_session)
        self.messages = SimpleNamespace(append_batch=self.append_batch)
        self.created, self.batched, self.closed = [], [], False

    def create_session(self, session):
        self.created.append(session)

    def append_batch(self, session_id, messages):
        self.batched.append((session_id, messages))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False


class LegacySessionDB:
    """A SessionDB with the batch verb, as the minimum Hermes carries."""

    def __init__(self, path):
        self.path = path
        self.created, self.batched, self.closed = [], [], False

    def create_session(self, session_id, source):
        self.created.append((session_id, source))

    def append_messages_batch(self, session_id, messages):
        self.batched.append((session_id, messages))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False


class BrokenModule(ModuleType):
    """A package that fails while the helper imports a name from it."""

    def __init__(self, name, error):
        super().__init__(name)
        self.error = error

    def __getattr__(self, item):
        raise self.error


class BrokenPersistence:
    """A finder that fails the import of the persistence package with a plain ImportError.

    A partial package can fail inside its own ``__init__`` this way, and that must stop the run.
    """

    def find_spec(self, name, path=None, target=None):
        if name == "persistence":
            raise ImportError("Synthetic partial package failure.", name="persistence")
        return None


class AbsentModules:
    """A finder that reports the given module names as absent, as a Hermes without them does."""

    def __init__(self, names):
        self.names = frozenset(names)

    def find_spec(self, name, path=None, target=None):
        if name in self.names:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None


def module(name, **attrs):
    made = ModuleType(name)
    for key, value in attrs.items():
        setattr(made, key, value)
    return made


@contextmanager
def modules_loaded(entries, finder=None):
    """Load ``entries`` as ``sys.modules`` and hide any real persistence package."""
    with patch.dict(sys.modules, entries):
        for key in [key for key in sys.modules if key == "persistence" or key.startswith("persistence.")]:
            if key not in entries:
                del sys.modules[key]
        if finder is None:
            yield
        else:
            with patch.object(sys, "meta_path", [finder, *sys.meta_path]):
                yield


def storage_modules(store, opened):
    return {
        "persistence": module("persistence"),
        "persistence.conversation": module("persistence.conversation"),
        "persistence.conversation.messages": module("persistence.conversation.messages", NewMessage=NewMessage),
        "persistence.conversation.sessions": module("persistence.conversation.sessions", NewSession=NewSession),
        "persistence_assembly": module(
            "persistence_assembly",
            open_persistence=lambda scope, eager=False: (opened.append((scope, eager)), store)[1],
            profile_scope=lambda home=None: SimpleNamespace(profile_dir=home)),
    }


class SeedTranscriptTest(unittest.TestCase):
    def fallback_stub(self):
        """A SessionDB stub that fails the test if the old path runs."""
        return module("hermes_state", SessionDB=lambda path: self.fail("The fallback ran."))

    def test_writes_through_the_storage_interface_when_hermes_has_it(self):
        store, opened = Store(), []
        with modules_loaded(storage_modules(store, opened)):
            harness.seed_transcript(Path("run") / "home", "synthetic-session", [ROW, SUMMARY_ROW])
        self.assertEqual([session.fields for session in store.created],
                         [{"session_id": "synthetic-session", "source": "cli"}])
        self.assertEqual(store.batched[0][0], "synthetic-session")
        self.assertEqual([message.fields for message in store.batched[0][1]],
                         [{"role": "user", "content": ROW["content"]},
                          {"role": "user", "content": SUMMARY_ROW["content"],
                           "compressed_summary": True, "display_kind": "compaction_notice"}])
        self.assertEqual([(scope.profile_dir, eager) for scope, eager in opened], [(Path("run") / "home", True)])
        self.assertTrue(store.closed)

    def test_uses_the_batch_verb_when_the_package_is_absent(self):
        legacy = LegacySessionDB(Path("run") / "home" / "state.db")
        entries = {"hermes_state": module("hermes_state", SessionDB=lambda path: legacy)}
        with modules_loaded(entries, finder=AbsentModules({"persistence"})):
            harness.seed_transcript(Path("run") / "home", "synthetic-session", [ROW])
        self.assertEqual(legacy.path, Path("run") / "home" / "state.db")
        self.assertEqual(legacy.created, [("synthetic-session", "cli")])
        self.assertEqual(legacy.batched, [("synthetic-session", [ROW])])
        self.assertTrue(legacy.closed)

    def test_a_package_that_fails_on_import_stops_the_run(self):
        entries = {"hermes_state": self.fallback_stub()}
        with modules_loaded(entries, finder=BrokenPersistence()):
            with self.assertRaises(ImportError):
                harness.seed_transcript(Path("run") / "home", "synthetic-session", [ROW])

    def test_a_package_without_the_conversation_part_fails(self):
        entries = {"persistence": module("persistence"), "hermes_state": self.fallback_stub()}
        with modules_loaded(entries):
            with self.assertRaises(ModuleNotFoundError):
                harness.seed_transcript(Path("run") / "home", "synthetic-session", [ROW])

    def test_a_package_without_the_assembly_module_fails(self):
        entries = {key: value for key, value in storage_modules(Store(), []).items()
                   if key != "persistence_assembly"}
        entries["hermes_state"] = self.fallback_stub()
        with modules_loaded(entries, finder=AbsentModules({"persistence_assembly"})):
            with self.assertRaises(ModuleNotFoundError):
                harness.seed_transcript(Path("run") / "home", "synthetic-session", [ROW])

    def test_a_missing_third_party_module_inside_the_storage_interface_fails(self):
        store, opened = Store(), []
        entries = storage_modules(store, opened)
        entries["persistence_assembly"] = BrokenModule("persistence_assembly",
                                                       ModuleNotFoundError("No module named 'sqlalchemy'",
                                                                           name="sqlalchemy"))
        entries["hermes_state"] = self.fallback_stub()
        with modules_loaded(entries):
            with self.assertRaises(ModuleNotFoundError):
                harness.seed_transcript(Path("run") / "home", "synthetic-session", [ROW])
        self.assertEqual(opened, [])
        self.assertFalse(store.closed)

    def test_a_missing_name_in_the_storage_package_fails(self):
        store, opened = Store(), []
        entries = storage_modules(store, opened)
        entries["persistence_assembly"] = BrokenModule(
            "persistence_assembly",
            ImportError("cannot import name 'open_persistence' from 'persistence_assembly'",
                        name="persistence_assembly"))
        entries["hermes_state"] = self.fallback_stub()
        with modules_loaded(entries):
            with self.assertRaises(ImportError):
                harness.seed_transcript(Path("run") / "home", "synthetic-session", [ROW])
        self.assertEqual(opened, [])

    def test_an_error_inside_the_storage_interface_fails(self):
        store, opened = Store(), []
        entries = storage_modules(store, opened)
        entries["persistence_assembly"] = BrokenModule("persistence_assembly", RuntimeError("Synthetic failure."))
        with modules_loaded(entries):
            with self.assertRaises(RuntimeError):
                harness.seed_transcript(Path("run") / "home", "synthetic-session", [ROW])
        self.assertEqual(opened, [])


if __name__ == "__main__":
    unittest.main()
