"""A candidate above the compaction threshold that fits the window is committed, not refused."""

import unittest

from wc_fixtures import assistant, user

# The synthetic route: a 64 000-token window, the smallest allowed threshold fraction (0.10, 6 400 tokens),
# a captured system prompt of about 4 944 tokens, and one indivisible native replay block. Below the
# threshold about 950 tokens stay free: less than the smallest candidate.
WINDOW = 64_000
THRESHOLD = 0.10
SYSTEM_LINE = "Synthetic system prompt line for a fake route. "
SYSTEM_LINES = 420
REPLAY_CHARS = 20_000


class ThresholdPolicyTest(unittest.TestCase):
    def setUp(self):
        from test_wc_engine import EngineTest
        fixture = EngineTest("test_threshold_comes_from_the_setting")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.engine = fixture.make(warm=False, threshold=THRESHOLD)
        self.engine.update_model(model="fake", context_length=WINDOW, base_url="http://127.0.0.1:9/v1",
                                 api_key="fake", provider="custom", api_mode="codex_responses")

    def history(self, replay_chars):
        """Return the stored rows and the captured rows: synthetic, with one indivisible replay block."""
        from test_wc_engine import old_turns
        rows = [*old_turns(3), user("go")]
        reply = assistant("synthetic", codex_reasoning_items=[
            {"type": "reasoning", "encrypted_content": "x" * replay_chars}])
        return rows, [*rows, reply]

    def captured_request(self, rows, lines=SYSTEM_LINES):
        """Return the capture of a codex_responses request: a large system prompt and the stored rows."""
        from warm_compaction.responses import wire_rows
        from warm_compaction.rows import row_digest
        body = {"model": "fake", "instructions": (SYSTEM_LINE * lines).strip(), "store": False,
                "input": wire_rows(rows), "max_output_tokens": 4_096}
        capture = {"route": self.engine._wc_route, "digests": [row_digest(row) for row in rows], "body": body,
                   "reply": {"role": "assistant", "content": "synthetic", "tool_calls": []}}
        self.fixture.store.latest = lambda session_id, value=capture: value
        return capture

    def budget(self, capture, history):
        """Return the captured overhead, the reply reserve, and the free room below the threshold."""
        from warm_compaction.engine import request_overhead, request_reserve
        overhead = request_overhead(capture, history)
        self.assertIsNotNone(overhead, "the capture supplies the overhead")
        reserve = request_reserve(capture, WINDOW)
        free = self.engine._room([], "", int(overhead), reserve)
        self.assertIsNotNone(free, "a threshold gives a free room")
        return int(overhead), reserve, int(free)

    def test_a_candidate_above_the_threshold_that_fits_the_window_is_committed(self):
        # Before the fix the final check compared the candidate to the threshold only: the history fits the
        # window, but the compaction refused with fallback=capacity and the turn failed.
        from warm_compaction.rows import sent_rows, sent_tokens, estimate_tokens
        rows, history = self.history(REPLAY_CHARS)
        before = list(history)
        capture = self.captured_request(rows)
        overhead, reserve, free = self.budget(capture, history)
        policy = self.engine._policy(history)
        self.assertGreater(free, 0, "a summary still has room below the threshold")
        self.assertGreater(sent_tokens(history[-1], policy), free, "the replay block is larger than the room")
        self.assertLessEqual(sent_tokens(history[-1], policy), WINDOW - reserve - overhead,
                            "the replay block fits the window")
        new = self.engine.compress(history)
        self.assertIsNot(new, history)
        self.assertEqual(history, before, "the input history is unchanged")
        self.assertEqual(self.engine.compression_count, 1)
        candidate = estimate_tokens(sent_rows(new, policy))
        above = int((self.engine.warm_last or {}).get("over_threshold") or 0)
        self.assertEqual(above, overhead + candidate - self.engine.threshold_tokens,
                         "the status gives the distance above the threshold")
        self.assertGreater(above, 0)
        self.assertGreater(candidate + overhead, self.engine.threshold_tokens, "the candidate stays above the "
                            "threshold: the threshold is a policy")
        self.assertLessEqual(candidate, WINDOW - reserve - overhead, "the candidate fits the window")
        self.assertEqual((self.engine.warm_last or {}).get("path"), "fallback")

    def test_a_free_room_below_the_summary_structure_still_stops_before_the_candidate(self):
        # The documented exception: when the overhead leaves less room than the summary row structure, the
        # pass spends one summary request and then stops with summary_too_large. The window is not the limit
        # here, because the plugin cannot build a summary row in that room.
        from warm_compaction.rows import sent_tokens
        rows, history = self.history(REPLAY_CHARS)
        capture = self.captured_request(rows, lines=460)
        overhead, reserve, free = self.budget(capture, history)
        self.assertGreater(free, 0, "the free room is still positive")
        self.assertGreater(sent_tokens(history[-1], self.engine._policy(history)), free)
        self.fixture.assert_aborted(self.engine, history, "summary_too_large")
        self.assertEqual(self.engine.compression_count, 0)

    def test_a_candidate_inside_the_summary_structure_allowance_is_committed_without_a_warning(self):
        # The free room subtracts the summary row structure. A candidate above that room can still stay below
        # the threshold: it is committed, and the status has no over_threshold field.
        from warm_compaction.rows import estimate_tokens, sent_rows
        rows, history = self.history(400)
        capture = self.captured_request(rows)
        overhead, reserve, free = self.budget(capture, history)
        new = self.engine.compress(history)
        self.assertIsNot(new, history)
        candidate = estimate_tokens(sent_rows(new, self.engine._policy(new)))
        self.assertGreater(candidate, free, "the candidate is above the free room")
        self.assertLessEqual(overhead + candidate, self.engine.threshold_tokens,
                             "the candidate stays below the threshold")
        self.assertEqual((self.engine.warm_last or {}).get("path"), "fallback")
        self.assertNotIn("over_threshold", self.engine.warm_last or {})

    def test_a_candidate_above_the_window_is_still_refused(self):
        # The window less the reply reserve and the overhead stays the hard limit. A replay block above it
        # cannot be sent at all: keep the history.
        from warm_compaction.rows import sent_tokens
        rows, history = self.history(300_000)
        capture = self.captured_request(rows)
        overhead, reserve, _free = self.budget(capture, history)
        policy = self.engine._policy(history)
        self.assertGreater(sent_tokens(history[-1], policy), WINDOW - reserve - overhead,
                           "the replay block is above the window")
        self.fixture.assert_aborted(self.engine, history, "capacity")
        self.assertEqual(self.engine.compression_count, 0)


if __name__ == "__main__":
    unittest.main()
