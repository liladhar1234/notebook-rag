"""QueryPlanner: follow-up rewriting + multi-query in one LLM call, with safe fallbacks."""
from __future__ import annotations

import json

from notebook_rag.retrieval.planner import QueryPlanner


class ScriptedLLM:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def complete(self, system, user):
        self.prompts.append(user)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def reply(standalone, queries=None):
    body = {"standalone_question": standalone}
    if queries is not None:
        body["search_queries"] = queries
    return json.dumps(body)


def test_no_history_and_no_expansion_skips_the_llm():
    llm = ScriptedLLM(reply("x"))
    plan = QueryPlanner(llm, num_variants=0).plan("what is kafka")
    assert llm.prompts == [] and plan.queries == ["what is kafka"] and not plan.rewritten


def test_follow_up_is_rewritten_using_history():
    llm = ScriptedLLM(reply("What does the Alpha plan cost?"))
    history = [("user", "What does the Beta plan cost?"), ("assistant", "Beta costs 40 dollars.")]
    plan = QueryPlanner(llm, num_variants=0).plan("and the Alpha one?", history)
    assert plan.standalone == "What does the Alpha plan cost?" and plan.rewritten
    assert plan.queries == ["What does the Alpha plan cost?", "and the Alpha one?"]  # user's own words kept
    assert "User: What does the Beta plan cost?" in llm.prompts[0]
    assert "Assistant: Beta costs 40 dollars." in llm.prompts[0]
    assert "search_queries" not in llm.prompts[0]  # expansion off: only the rewrite is requested


def test_variants_are_deduplicated_and_capped():
    llm = ScriptedLLM(reply("big data skills", ["Big Data Skills", "hadoop spark experience", "kafka", "extra one", "another"]))
    plan = QueryPlanner(llm, num_variants=2).plan("big data skills")
    assert plan.queries == ["big data skills", "hadoop spark experience", "kafka"]
    assert plan.variants == 2 and not plan.rewritten


def test_json_inside_fences_or_prose_is_parsed():
    raw = 'Sure! ```json\n' + reply("clean question", ["alt wording here"]) + '\n```'
    plan = QueryPlanner(ScriptedLLM(raw), num_variants=1).plan("messy q")
    assert plan.standalone == "clean question" and "alt wording here" in plan.queries


def test_garbage_reply_falls_back_to_the_original_question():
    plan = QueryPlanner(ScriptedLLM("I cannot do that"), num_variants=2).plan("q one")
    assert plan.queries == ["q one"] and plan.standalone == "q one"


def test_llm_error_falls_back_to_the_original_question():
    plan = QueryPlanner(ScriptedLLM(RuntimeError("quota")), num_variants=2).plan("q one", [("user", "hi")])
    assert plan.queries == ["q one"] and not plan.rewritten


def test_unusable_standalone_is_ignored():
    for bad in ("", "   ", "x" * 600):
        plan = QueryPlanner(ScriptedLLM(reply(bad)), num_variants=0).plan("real q", [("user", "a")])
        assert plan.standalone == "real q"


def test_non_string_variants_are_ignored_and_long_history_is_clipped():
    llm = ScriptedLLM(reply("q", [1, None, "valid variant"]))
    plan = QueryPlanner(llm, num_variants=3).plan("q", [("assistant", "z " * 1000)])
    assert plan.queries == ["q", "valid variant"]
    assert len(llm.prompts[0]) < 2000  # the 2000-char turn was clipped
