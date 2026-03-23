"""
Unit tests for layout_gen.synth.geo — geometric DRC fix agent.

Tests cover:
- Rect properties and spatial queries (LayoutState)
- Action application (StretchEdge, MoveShape, AddRect, RemoveShape, MergeShapes)
- Violation parsing (category detection, layer normalization, value extraction)
- RuleGeoAgent fix proposals (spacing, width, enclosure, area, overlap)
- LearnedGeoAgent fallback behaviour
- ReplayBuffer mechanics
- Observation encoding
"""
from __future__ import annotations

import pytest
import numpy as np

from layout_gen.drc.base import DRCViolation
from layout_gen.synth.geo.state import Rect, LayoutState
from layout_gen.synth.geo.actions import (
    StretchEdge, MoveShape, AddRect, RemoveShape, MergeShapes,
    apply_action,
)
from layout_gen.synth.geo.violations import (
    ViolationInfo, parse_violation, parse_violations,
    _normalize_layer, _detect_category, _extract_required,
)
from layout_gen.synth.geo.agent import RuleGeoAgent
from layout_gen.synth.geo.learned_agent import (
    LearnedGeoAgent, ReplayBuffer, Transition, Observation,
    encode_observation,
)


# ── Rect tests ───────────────────────────────────────────────────────────────

class TestRect:
    def test_properties(self):
        r = Rect(0, "met1", 1.0, 2.0, 3.0, 5.0)
        assert r.width == pytest.approx(2.0)
        assert r.height == pytest.approx(3.0)
        assert r.area == pytest.approx(6.0)
        assert r.cx == pytest.approx(2.0)
        assert r.cy == pytest.approx(3.5)

    def test_overlaps_true(self):
        a = Rect(0, "met1", 0, 0, 2, 2)
        b = Rect(1, "met1", 1, 1, 3, 3)
        assert a.overlaps(b)
        assert b.overlaps(a)

    def test_overlaps_false(self):
        a = Rect(0, "met1", 0, 0, 1, 1)
        b = Rect(1, "met1", 2, 2, 3, 3)
        assert not a.overlaps(b)

    def test_overlaps_shared_edge(self):
        a = Rect(0, "met1", 0, 0, 1, 1)
        b = Rect(1, "met1", 1, 0, 2, 1)
        assert a.overlaps(b)

    def test_edge_dist_separated(self):
        a = Rect(0, "met1", 0, 0, 1, 1)
        b = Rect(1, "met1", 2, 0, 3, 1)  # 1 µm gap in X
        assert a.edge_dist(b) == pytest.approx(1.0)

    def test_edge_dist_overlapping(self):
        a = Rect(0, "met1", 0, 0, 2, 2)
        b = Rect(1, "met1", 1, 1, 3, 3)
        assert a.edge_dist(b) == pytest.approx(0.0)

    def test_edge_dist_diagonal(self):
        a = Rect(0, "met1", 0, 0, 1, 1)
        b = Rect(1, "met1", 2, 2, 3, 3)
        # Diagonal: dx=1, dy=1 → sqrt(2)
        assert a.edge_dist(b) == pytest.approx(2**0.5, rel=1e-3)

    def test_contains_point(self):
        r = Rect(0, "met1", 0, 0, 2, 2)
        assert r.contains_point(1, 1)
        assert r.contains_point(0, 0)
        assert not r.contains_point(3, 3)

    def test_copy(self):
        r = Rect(0, "met1", 0, 0, 1, 1)
        c = r.copy()
        assert c.rid == r.rid
        assert c.x0 == r.x0
        c.x0 = 5
        assert r.x0 == 0  # original unchanged


# ── LayoutState tests ─────────────────────────────────────────────────────────

class TestLayoutState:
    def test_add_and_len(self):
        s = LayoutState()
        s.add("met1", 0, 0, 1, 1)
        s.add("met2", 2, 2, 3, 3)
        assert len(s) == 2

    def test_add_normalizes_coords(self):
        s = LayoutState()
        r = s.add("met1", 3, 4, 1, 2)  # x0>x1, y0>y1
        assert r.x0 == 1 and r.x1 == 3
        assert r.y0 == 2 and r.y1 == 4

    def test_remove(self):
        s = LayoutState()
        r = s.add("met1", 0, 0, 1, 1)
        removed = s.remove(r.rid)
        assert removed is not None
        assert len(s) == 0
        assert s.remove(999) is None

    def test_update(self):
        s = LayoutState()
        r = s.add("met1", 0, 0, 1, 1)
        s.update(r.rid, x1=2.0)
        assert s[r.rid].x1 == pytest.approx(2.0)

    def test_update_auto_normalizes(self):
        s = LayoutState()
        r = s.add("met1", 0, 0, 1, 1)
        s.update(r.rid, x0=5.0)  # x0 > x1 → swap
        assert s[r.rid].x0 == pytest.approx(1.0)
        assert s[r.rid].x1 == pytest.approx(5.0)

    def test_contains_and_getitem(self):
        s = LayoutState()
        r = s.add("met1", 0, 0, 1, 1)
        assert r.rid in s
        assert s[r.rid].layer == "met1"
        assert 999 not in s

    def test_iter(self):
        s = LayoutState()
        s.add("met1", 0, 0, 1, 1)
        s.add("met2", 2, 2, 3, 3)
        layers = {r.layer for r in s}
        assert layers == {"met1", "met2"}

    def test_on_layer(self):
        s = LayoutState()
        s.add("met1", 0, 0, 1, 1)
        s.add("met1", 2, 2, 3, 3)
        s.add("met2", 4, 4, 5, 5)
        assert len(s.on_layer("met1")) == 2
        assert len(s.on_layer("met2")) == 1
        assert len(s.on_layer("poly")) == 0

    def test_near(self):
        s = LayoutState()
        s.add("met1", 0, 0, 1, 1)      # centre (0.5, 0.5)
        s.add("met1", 10, 10, 11, 11)   # centre (10.5, 10.5) — far
        result = s.near(0.5, 0.5, radius=2.0)
        assert len(result) == 1

    def test_near_layer_filter(self):
        s = LayoutState()
        s.add("met1", 0, 0, 1, 1)
        s.add("met2", 0, 0, 1, 1)
        assert len(s.near(0.5, 0.5, 2.0, layer="met1")) == 1

    def test_at_point(self):
        s = LayoutState()
        s.add("met1", 0, 0, 2, 2)
        s.add("met1", 5, 5, 6, 6)
        assert len(s.at_point(1, 1)) == 1
        assert len(s.at_point(5.5, 5.5)) == 1
        assert len(s.at_point(3, 3)) == 0

    def test_neighbours(self):
        s = LayoutState()
        r0 = s.add("met1", 0, 0, 1, 1)
        s.add("met1", 1.5, 0, 2.5, 1)   # 0.5 µm gap
        s.add("met1", 10, 10, 11, 11)    # far
        nbrs = s.neighbours(r0.rid, max_dist=1.0)
        assert len(nbrs) == 1
        assert nbrs[0][1] == pytest.approx(0.5)

    def test_spacing_pairs(self):
        s = LayoutState()
        s.add("met1", 0, 0, 1, 1)
        s.add("met1", 1.2, 0, 2.2, 1)   # 0.2 µm gap
        s.add("met1", 5, 5, 6, 6)        # too far
        pairs = s.spacing_pairs("met1", max_dist=0.5)
        assert len(pairs) == 1
        assert pairs[0][2] == pytest.approx(0.2)

    def test_init_from_rects(self):
        rects = [
            Rect(0, "met1", 0, 0, 1, 1),
            Rect(5, "met2", 2, 2, 3, 3),
        ]
        s = LayoutState(rects)
        assert len(s) == 2
        assert 0 in s
        assert 5 in s
        # Next rid should be > max existing
        r = s.add("poly", 0, 0, 1, 1)
        assert r.rid >= 6

    def test_local_crop(self):
        s = LayoutState()
        s.add("met1", 0, 0, 1, 1)
        s.add("met2", 0, 0, 0.5, 0.5)
        s.add("met1", 10, 10, 11, 11)  # far
        crop = s.local_crop(0.5, 0.5, radius=2.0)
        assert crop.shape == (2, 6)
        assert crop.dtype == np.float32
        # Coordinates should be relative to (0.5, 0.5)
        # x0_rel, y0_rel should be negative for shapes starting at 0
        for row in crop:
            assert row[1] <= 0  # x0_rel


# ── Action tests ──────────────────────────────────────────────────────────────

class TestActions:
    def _make_state(self):
        s = LayoutState()
        s.add("met1", 0, 0, 1, 1)  # rid=0
        s.add("met2", 2, 2, 4, 4)  # rid=1
        return s

    def test_stretch_right(self):
        s = self._make_state()
        apply_action(s, StretchEdge(0, "right", 0.5))
        assert s[0].x1 == pytest.approx(1.5)
        assert s[0].x0 == pytest.approx(0)  # unchanged

    def test_stretch_left(self):
        s = self._make_state()
        apply_action(s, StretchEdge(0, "left", 0.3))
        assert s[0].x0 == pytest.approx(-0.3)

    def test_stretch_top(self):
        s = self._make_state()
        apply_action(s, StretchEdge(0, "top", 0.2))
        assert s[0].y1 == pytest.approx(1.2)

    def test_stretch_bottom(self):
        s = self._make_state()
        apply_action(s, StretchEdge(0, "bottom", 0.1))
        assert s[0].y0 == pytest.approx(-0.1)

    def test_move_shape(self):
        s = self._make_state()
        apply_action(s, MoveShape(0, dx=1.0, dy=-0.5))
        assert s[0].x0 == pytest.approx(1.0)
        assert s[0].x1 == pytest.approx(2.0)
        assert s[0].y0 == pytest.approx(-0.5)
        assert s[0].y1 == pytest.approx(0.5)

    def test_add_rect(self):
        s = self._make_state()
        r = apply_action(s, AddRect("poly", 5, 5, 6, 6))
        assert len(s) == 3
        assert r.layer == "poly"

    def test_remove_shape(self):
        s = self._make_state()
        apply_action(s, RemoveShape(0))
        assert 0 not in s
        assert len(s) == 1

    def test_merge_shapes(self):
        s = LayoutState()
        s.add("met1", 0, 0, 1, 1)   # rid=0
        s.add("met1", 0.5, 0, 2, 1) # rid=1, overlapping
        merged = apply_action(s, MergeShapes(rids=[0, 1]))
        assert len(s) == 1
        assert merged.x0 == pytest.approx(0)
        assert merged.x1 == pytest.approx(2)
        assert merged.layer == "met1"

    def test_merge_empty(self):
        s = LayoutState()
        result = apply_action(s, MergeShapes(rids=[99, 100]))
        assert result is None

    def test_unknown_action_raises(self):
        s = LayoutState()
        with pytest.raises(TypeError):
            apply_action(s, "not_an_action")

    def test_describe(self):
        assert "right" in StretchEdge(0, "right", 0.1).describe()
        assert "Move" in MoveShape(0, 1, 0).describe()
        assert "Add" in AddRect("met1", 0, 0, 1, 1).describe()
        assert "Remove" in RemoveShape(0).describe()
        assert "Merge" in MergeShapes([0, 1]).describe()


# ── Violation parsing tests ───────────────────────────────────────────────────

class TestViolationParsing:
    def test_normalize_layer(self):
        assert _normalize_layer("m1") == "met1"
        assert _normalize_layer("M2") == "met2"
        assert _normalize_layer("li") == "li1"
        assert _normalize_layer("active") == "diff"
        assert _normalize_layer("gate") == "poly"
        assert _normalize_layer("contact") == "licon1"
        assert _normalize_layer("nw") == "nwell"
        assert _normalize_layer("met1") == "met1"

    def test_detect_category_spacing(self):
        assert _detect_category("met1.2", "spacing < 0.14 um") == "spacing"
        assert _detect_category("li.sp", "space violation") == "spacing"

    def test_detect_category_width(self):
        assert _detect_category("met2.1", "minimum width < 0.14 um") == "width"
        assert _detect_category("li.wid", "width violation") == "width"

    def test_detect_category_enclosure(self):
        assert _detect_category("m1.5", "enclosure of via1") == "enclosure"

    def test_detect_category_area(self):
        assert _detect_category("li.6", "min area < 0.0561 um") == "area"

    def test_detect_category_overlap(self):
        assert _detect_category("met1.3", "short circuit") == "overlap"

    def test_detect_category_unknown(self):
        assert _detect_category("foo.bar", "something else") == "unknown"

    def test_extract_required(self):
        assert _extract_required("spacing < 0.14 um") == pytest.approx(0.14)
        assert _extract_required("width: 0.17µm") == pytest.approx(0.17)
        assert _extract_required("no value here") == pytest.approx(0.0)

    def test_parse_violation_spacing(self):
        v = DRCViolation(
            rule="met2.2",
            description="spacing < 0.14 um",
            x=1.5, y=2.5,
            value=0.10,
        )
        info = parse_violation(v)
        assert info.category == "spacing"
        assert info.layer == "met2"
        assert info.required == pytest.approx(0.14)
        assert info.measured == pytest.approx(0.10)
        assert info.deficit == pytest.approx(0.04)
        assert info.x == pytest.approx(1.5)

    def test_parse_violation_enclosure(self):
        v = DRCViolation(
            rule="m1.5",
            description="enclosure of via1 by met1 < 0.085 um",
            x=0, y=0,
            value=0.05,
        )
        info = parse_violation(v)
        assert info.category == "enclosure"
        assert info.layer == "met1"
        assert info.inner_layer == "via1"
        assert info.deficit == pytest.approx(0.035)

    def test_parse_violation_no_measured(self):
        v = DRCViolation(
            rule="li.6",
            description="min area < 0.0561 um",
            x=0, y=0,
            value=None,
        )
        info = parse_violation(v)
        assert info.category == "area"
        assert info.measured == 0.0
        assert info.deficit == pytest.approx(0.0561)  # required - measured

    def test_parse_violations_batch(self):
        vs = [
            DRCViolation("met1.2", "spacing < 0.14 um", x=0, y=0, value=0.1),
            DRCViolation("li.1", "width < 0.17 um", x=1, y=1, value=0.12),
        ]
        infos = parse_violations(vs)
        assert len(infos) == 2
        assert infos[0].category == "spacing"
        assert infos[1].category == "width"


# ── RuleGeoAgent tests ────────────────────────────────────────────────────────

class TestRuleGeoAgent:
    def test_fix_spacing(self):
        """Two met1 shapes too close → agent proposes MoveShape."""
        s = LayoutState()
        s.add("met1", 0, 0, 1, 1)
        s.add("met1", 1.05, 0, 2.05, 1)  # 0.05 µm gap, need 0.14

        v = ViolationInfo(
            category="spacing", layer="met1",
            measured=0.05, required=0.14, deficit=0.09,
            x=1.025, y=0.5,
        )
        agent = RuleGeoAgent(search_radius=3.0)
        actions = agent.propose_fix(s, v)
        assert len(actions) == 1
        assert isinstance(actions[0], MoveShape)

    def test_fix_width(self):
        """Narrow met1 shape → agent proposes StretchEdge pair."""
        s = LayoutState()
        s.add("met1", 0, 0, 0.10, 1)  # 0.10 µm wide, need 0.14

        v = ViolationInfo(
            category="width", layer="met1",
            measured=0.10, required=0.14, deficit=0.04,
            x=0.05, y=0.5,
        )
        agent = RuleGeoAgent(search_radius=3.0)
        actions = agent.propose_fix(s, v)
        assert len(actions) == 2
        assert all(isinstance(a, StretchEdge) for a in actions)
        edges = {a.edge for a in actions}
        assert "left" in edges
        assert "right" in edges

    def test_fix_enclosure(self):
        """Via not enclosed enough → agent stretches outer shape."""
        s = LayoutState()
        # Inner (via1) shape
        s.add("via1", 1.0, 1.0, 1.15, 1.15)
        # Outer (met1) shape — not enough enclosure on left
        s.add("met1", 0.98, 0.9, 1.3, 1.3)

        v = ViolationInfo(
            category="enclosure", layer="met1", inner_layer="via1",
            measured=0.02, required=0.085, deficit=0.065,
            x=1.0, y=1.0,
        )
        agent = RuleGeoAgent(search_radius=3.0)
        actions = agent.propose_fix(s, v)
        assert len(actions) > 0
        assert all(isinstance(a, StretchEdge) for a in actions)

    def test_fix_area(self):
        """Small met1 shape → agent enlarges it."""
        s = LayoutState()
        s.add("met1", 0, 0, 0.2, 0.2)  # area=0.04, need 0.083

        v = ViolationInfo(
            category="area", layer="met1",
            measured=0.04, required=0.083, deficit=0.043,
            x=0.1, y=0.1,
        )
        agent = RuleGeoAgent(search_radius=3.0)
        actions = agent.propose_fix(s, v)
        assert len(actions) == 2
        assert all(isinstance(a, StretchEdge) for a in actions)

    def test_fix_overlap(self):
        """Two overlapping met1 shapes → agent moves one."""
        s = LayoutState()
        s.add("met1", 0, 0, 1, 1)
        s.add("met1", 0.5, 0.5, 1.5, 1.5)

        v = ViolationInfo(
            category="overlap", layer="met1",
            x=0.75, y=0.75,
        )
        agent = RuleGeoAgent(search_radius=3.0)
        actions = agent.propose_fix(s, v)
        assert len(actions) >= 1
        assert isinstance(actions[0], MoveShape)

    def test_fix_unknown_fallback(self):
        """Unknown category → agent tries spacing then width."""
        s = LayoutState()
        s.add("met1", 0, 0, 1, 1)
        s.add("met1", 1.05, 0, 2, 1)

        v = ViolationInfo(
            category="unknown", layer="met1",
            measured=0.05, required=0.14, deficit=0.09,
            x=1.0, y=0.5,
        )
        agent = RuleGeoAgent(search_radius=3.0)
        actions = agent.propose_fix(s, v)
        assert len(actions) > 0

    def test_fix_batch(self):
        """fix_batch handles multiple violations."""
        s = LayoutState()
        s.add("met1", 0, 0, 0.10, 1)  # narrow
        s.add("met1", 5, 0, 5.10, 1)  # narrow

        vs = [
            ViolationInfo(category="width", layer="met1",
                          measured=0.10, required=0.14, deficit=0.04,
                          x=0.05, y=0.5),
            ViolationInfo(category="width", layer="met1",
                          measured=0.10, required=0.14, deficit=0.04,
                          x=5.05, y=0.5),
        ]
        agent = RuleGeoAgent(search_radius=3.0)
        actions = agent.fix_batch(s, vs)
        assert len(actions) == 4  # 2 stretches per violation

    def test_no_shapes_returns_empty(self):
        """Agent returns empty list if no shapes found."""
        s = LayoutState()
        v = ViolationInfo(category="spacing", layer="met1",
                          measured=0.05, required=0.14, deficit=0.09,
                          x=5, y=5)
        agent = RuleGeoAgent(search_radius=1.0)
        assert agent.propose_fix(s, v) == []

    def test_repulsion_vector_horizontal(self):
        """Repulsion pushes in direction of shortest separation."""
        a = Rect(0, "met1", 0, 0, 1, 1)
        b = Rect(1, "met1", 1.5, 0, 2.5, 1)  # to the right, 0.5 gap
        dx, dy = RuleGeoAgent._repulsion_vector(a, b, 0.8)
        # Should push a leftward (negative dx)
        assert dx < 0
        assert dy == pytest.approx(0.0)


# ── LearnedGeoAgent tests ────────────────────────────────────────────────────

class TestLearnedGeoAgent:
    def test_fallback_to_rule_agent(self):
        """Untrained LearnedGeoAgent falls back to rule-based agent."""
        s = LayoutState()
        s.add("met1", 0, 0, 0.10, 1)

        v = ViolationInfo(
            category="width", layer="met1",
            measured=0.10, required=0.14, deficit=0.04,
            x=0.05, y=0.5,
        )
        fallback = RuleGeoAgent(search_radius=3.0)
        agent = LearnedGeoAgent(fallback=fallback)
        assert not agent.is_trained
        actions = agent.propose_fix(s, v)
        assert len(actions) == 2  # from fallback

    def test_no_fallback_returns_empty(self):
        agent = LearnedGeoAgent()
        s = LayoutState()
        v = ViolationInfo(category="width", layer="met1",
                          x=0, y=0)
        assert agent.propose_fix(s, v) == []


class TestReplayBuffer:
    def _dummy_transition(self, reward=1.0):
        obs = Observation(
            layout_crop=np.zeros((3, 6), dtype=np.float32),
            violation_feat=np.zeros(8, dtype=np.float32),
        )
        return Transition(obs, action_idx=0, action_param=0.1,
                          target_rid=0, reward=reward, done=False)

    def test_push_and_len(self):
        buf = ReplayBuffer(capacity=100)
        assert len(buf) == 0
        buf.push(self._dummy_transition())
        assert len(buf) == 1

    def test_capacity_overflow(self):
        buf = ReplayBuffer(capacity=5)
        for i in range(10):
            buf.push(self._dummy_transition(reward=float(i)))
        assert len(buf) == 5

    def test_sample(self):
        buf = ReplayBuffer(capacity=100)
        for i in range(20):
            buf.push(self._dummy_transition())
        batch = buf.sample(5)
        assert len(batch) == 5
        assert all(isinstance(t, Transition) for t in batch)

    def test_record_experience(self):
        agent = LearnedGeoAgent()
        obs = Observation(
            layout_crop=np.zeros((2, 6), dtype=np.float32),
            violation_feat=np.zeros(8, dtype=np.float32),
        )
        agent.record_experience(obs, action_idx=1, action_param=0.05,
                                target_rid=3, reward=0.5, done=False)
        assert agent.replay_size == 1


class TestObservationEncoding:
    def test_encode_observation(self):
        s = LayoutState()
        s.add("met1", 0, 0, 1, 1)
        s.add("met2", 0, 0, 0.5, 0.5)

        v = ViolationInfo(
            category="spacing", layer="met1",
            measured=0.05, required=0.14, deficit=0.09,
            x=0.5, y=0.5,
        )
        obs = encode_observation(s, v, crop_radius=2.0)
        assert obs.layout_crop.shape[0] >= 1
        assert obs.layout_crop.shape[1] == 6
        assert obs.violation_feat.shape == (8,)
        # Category encoding: spacing=0
        assert obs.violation_feat[0] == pytest.approx(0.0)


# ── Integration: apply fix then verify geometry ──────────────────────────────

class TestIntegration:
    def test_spacing_fix_resolves_violation(self):
        """After applying spacing fix, the gap should be >= required."""
        s = LayoutState()
        r0 = s.add("met1", 0, 0, 1, 1)
        r1 = s.add("met1", 1.05, 0, 2.05, 1)  # 0.05 gap

        v = ViolationInfo(
            category="spacing", layer="met1",
            measured=0.05, required=0.14, deficit=0.09,
            x=1.025, y=0.5,
        )
        agent = RuleGeoAgent(search_radius=3.0)
        actions = agent.propose_fix(s, v)
        for a in actions:
            apply_action(s, a)

        # Verify gap is now >= required
        a, b = s[r0.rid], s[r1.rid]
        gap = a.edge_dist(b)
        assert gap >= 0.14

    def test_width_fix_resolves_violation(self):
        """After applying width fix, the shape should be >= required width."""
        s = LayoutState()
        r = s.add("met1", 0, 0, 0.10, 1)

        v = ViolationInfo(
            category="width", layer="met1",
            measured=0.10, required=0.14, deficit=0.04,
            x=0.05, y=0.5,
        )
        agent = RuleGeoAgent(search_radius=3.0)
        actions = agent.propose_fix(s, v)
        for a in actions:
            apply_action(s, a)

        fixed = s[r.rid]
        assert fixed.width >= 0.14
