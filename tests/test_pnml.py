"""PNML export.

A catalytic reaction network is a Petri net, and exporting one makes it readable by
that ecosystem. Three parts of the RAF model have no direct Place/Transition
equivalent, and the tests that matter are the ones checking each is handled explicitly:
catalysts become self-loops, alternative catalyst sets become separate transitions, and
food gets source transitions so it cannot run out.

Validated during development against **pm4py**, an independent PNML reader, which is
how the double-hyphen bug below was found. pm4py is AGPL and is deliberately not a
dependency; these tests use the standard library only.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from rafkit import parse_crs
from rafkit.pnml import to_pnml, write_pnml

NS = {"p": "http://www.pnml.org/version-2009/grammar/pnml"}


def _parse(net, **kw):
    return ET.fromstring(to_pnml(net, **kw))


def _ids(root, tag):
    return [e.get("id") for e in root.iterfind(f".//p:{tag}", NS)]


def _labels(root, tag):
    out = []
    for e in root.iterfind(f".//p:{tag}", NS):
        t = e.find("p:name/p:text", NS)
        out.append(t.text if t is not None else None)
    return out


class TestWellFormedness:
    def test_output_parses(self):
        net = parse_crs("Food: a, b\nr1 : a + b [c] => c\n")
        assert _parse(net).tag.endswith("pnml")

    def test_header_comment_has_no_double_hyphen(self):
        """Regression. The header is prepended as raw XML, so it bypasses
        ElementTree's escaping entirely, and `--` inside a comment makes the whole
        document unparseable. An independent PNML reader caught this; the export
        itself was silent."""
        net = parse_crs("Food: a, b\nr1 : a + b [c] {z} => c\nr2 : a [] => q\n")
        text = to_pnml(net)
        header = text[:text.index("<pnml")]
        assert "--" not in header.replace("<!--", "").replace("-->", "")
        ET.fromstring(text)          # would raise if it were not well-formed

    def test_every_arc_endpoint_exists(self):
        net = parse_crs("Food: a, b\nr1 : a + b [{c,d},e] => c\nr2 : a + c [c] => d\n")
        root = _parse(net)
        nodes = set(_ids(root, "place")) | set(_ids(root, "transition"))
        for arc in root.iterfind(".//p:arc", NS):
            assert arc.get("source") in nodes
            assert arc.get("target") in nodes


class TestMapping:
    def test_places_are_molecules(self):
        net = parse_crs("Food: a, b\nr1 : a + b [c] => c\n")
        assert set(_labels(_parse(net), "place")) == set(net.molecules)

    def test_alternative_catalyst_sets_become_separate_transitions(self):
        """A transition's preset is a conjunction, so it cannot express "either set"."""
        net = parse_crs("Food: a, b\nr1 : a + b [{c,d},e] => c\n")
        labels = [l for l in _labels(_parse(net), "transition")
                  if not l.startswith("source:")]
        assert sorted(labels) == ["r1", "r1#2"]

    def test_a_catalyst_becomes_a_self_loop(self):
        net = parse_crs("Food: a, b\nr1 : a + b [c] => c\n")
        root = _parse(net)
        place = {l: i for i, l in zip(_ids(root, "place"), _labels(root, "place"))}
        arcs = {(a.get("source"), a.get("target"))
                for a in root.iterfind(".//p:arc", NS)}
        # c catalyses r1: both directions must be present.
        assert (place["c"], "t0") in arcs and ("t0", place["c"]) in arcs

    def test_food_gets_source_transitions_so_it_cannot_run_out(self):
        net = parse_crs("Food: a, b\nr1 : a + b [c] => c\n")
        labels = _labels(_parse(net), "transition")
        assert "source:a" in labels and "source:b" in labels

    def test_food_sources_can_be_turned_off(self):
        net = parse_crs("Food: a, b\nr1 : a + b [c] => c\n")
        labels = _labels(_parse(net, food_sources=False), "transition")
        assert not any(l.startswith("source:") for l in labels)


class TestWhatCannotBeExpressed:
    def test_reactions_that_can_never_fire_are_omitted_and_counted(self):
        """chi = empty means "must be catalysed, and nothing does". Emitting it as an
        unconstrained transition would make it freely fireable, the opposite of the
        intent, so it is dropped and the header says how many."""
        net = parse_crs("Food: a, b\nr1 : a + b [c] => c\nr2 : a [] => q\n")
        text = to_pnml(net)
        labels = [l for l in _labels(ET.fromstring(text), "transition")
                  if not l.startswith("source:")]
        assert labels == ["r1"]
        assert "1 reaction(s) omitted" in text

    def test_inhibition_is_recorded_and_flagged_as_lossy(self):
        net = parse_crs("Food: a, b\nr1 : a + b [c] {z} => c\n")
        text = to_pnml(net)
        root = ET.fromstring(text)
        ts = root.find(".//p:transition/p:toolspecific", NS)
        assert ts is not None and ts.get("tool") == "rafkit"
        assert ts.find("p:inhibitors", NS).text == "z"
        assert "DIFFERENT system" in text     # the warning is not optional


def test_write_pnml_round_trips_through_a_file(tmp_path):
    net = parse_crs("Food: a, b\nr1 : a + b [c] => c\n")
    path = tmp_path / "net.pnml"
    write_pnml(net, path)
    assert ET.parse(path).getroot().tag.endswith("pnml")
