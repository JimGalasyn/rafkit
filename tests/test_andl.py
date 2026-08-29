"""ANDL export.

`rafkit.pnml` makes a network readable by the Petri net ecosystem; `rafkit.andl` makes
it runnable, which raises the stakes: a lossy readable file documents a different
system, a lossy runnable file EXECUTES one. The tests that matter are therefore the
refusals -- inhibition, catalyst-as-reactant -- and the semantic emissions: catalyst
self-loops, identical-reactant weight-2 arcs, named rate constants.

The Spike tests run only when the binary is present (SPIKE_BIN, or the default local
install); CI skips them. They validate the one semantic the 2026-08-29 three-toy
calibration did not cover -- an empty-preset source transition fires at constant
propensity -- and that generated files parse in the target tool at all.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from rafkit import parse_crs
from rafkit.andl import to_andl, write_andl
from rafkit.binary_polymer import BinaryPolymerNetwork


def _lines(text):
    return [ln.strip() for ln in text.splitlines()]


class TestMapping:
    def test_places_are_molecules_with_food_marked(self):
        net = parse_crs("Food: a, b\nr1 : a + b [c] => c\n")
        text = to_andl(net, 1.0)
        assert "s_1_a = 1;" in _lines(text)
        assert "s_1_b = 1;" in _lines(text)
        assert "s_1_c = 0;" in _lines(text)

    def test_catalyst_is_a_consume_and_produce_self_loop(self):
        net = parse_crs("Food: a, b\nr1 : a + b [z] => c\n")
        text = to_andl(net, 1.0)
        assert "[s_1_z - 1]" in text and "[s_1_z + 1]" in text

    def test_autocatalyst_merges_self_loop_with_product_arc(self):
        """c catalyses its own production: the self-loop's +1 and the product's +1
        aggregate to a single +2 arc. Pre-arc weight stays 1, so the propensity is
        k*A*B*C -- the autocatalytic loop rendered faithfully, not an error."""
        net = parse_crs("Food: a, b\nr1 : a + b [c] => c\n")
        text = to_andl(net, 1.0)
        assert "[s_1_c - 1]" in text and "[s_1_c + 2]" in text

    def test_alternative_catalyst_sets_become_separate_transitions(self):
        net = parse_crs("Food: a, b\nr1 : a + b [{y,z},w] => c\n")
        text = to_andl(net, 0.25)
        # Two catalytic channels, each with its own named constant at the same rate.
        assert text.count("MassAction(") == 2
        assert text.count("= 0.25;") == 2

    def test_conjunctive_catalyst_set_loops_every_member(self):
        net = parse_crs("Food: a, b\nr1 : a + b [{y,z}] => c\n")
        text = to_andl(net, 1.0)
        for m in ("s_1_y", "s_1_z"):
            assert f"[{m} - 1]" in text and f"[{m} + 1]" in text

    def test_identical_reactants_get_a_weight_two_arc(self):
        net = BinaryPolymerNetwork(molecules=("0", "00"), food=frozenset({0}),
                                   reactions=((0, 0, 1),), directions=(1,),
                                   catalysts=(frozenset(),), p=0.0, max_len=2,
                                   food_len=1)
        text = to_andl(net, 1.0, k_uncat=0.5)
        assert "[s_1_0 - 2]" in text

    def test_k_uncat_adds_a_background_channel_per_reaction(self):
        net = parse_crs("Food: a, b\nr1 : a + b [c] => c\n")
        text = to_andl(net, 1.0, k_uncat=0.05)
        assert "t_4_r1_u" in text
        assert "= 0.05;" in text
        # The background channel does not touch the catalyst.
        block = text[text.index("t_4_r1_u"):]
        block = block[:block.index(";")]
        assert "s_1_c" not in block

    def test_rates_are_named_constants_so_a_config_can_override_one(self):
        net = parse_crs("Food: a, b\nr1 : a + b [c] => c\n")
        text = to_andl(net, 0.125)
        assert "double k_t_2_r1 = 0.125;" in text
        assert "MassAction(k_t_2_r1)" in text

    def test_marking_overrides_and_food_default(self):
        net = parse_crs("Food: a, b\nr1 : a + b [c] => c\n")
        text = to_andl(net, 1.0, marking={"a": 40, "c": 7})
        assert "s_1_a = 40;" in _lines(text)
        assert "s_1_b = 1;" in _lines(text)      # food default survives elsewhere
        assert "s_1_c = 7;" in _lines(text)

    def test_influx_and_washout_transitions(self):
        net = parse_crs("Food: a, b\nr1 : a + b [c] => c\n")
        text = to_andl(net, 1.0, food_influx=40.0, washout={"c": 0.1})
        assert "t_5_src_a" in text and "t_5_src_b" in text
        assert "t_5_out_c" in text and "[s_1_c - 1]" in text


class TestRefusals:
    def test_inhibition_is_refused_not_dropped(self):
        net = parse_crs("Food: a, b\nr1 : a + b [c] {z} => c\n")
        with pytest.raises(ValueError, match="RUNS a different system"):
            to_andl(net, 1.0)

    def test_catalyst_that_is_also_a_reactant_is_refused(self):
        net = parse_crs("Food: a, b\nr1 : a + b [a] => c\n")
        with pytest.raises(ValueError, match="also reactants"):
            to_andl(net, 1.0)

    def test_k_is_required_and_length_checked(self):
        net = parse_crs("Food: a, b\nr1 : a + b [c] => c\n")
        with pytest.raises(ValueError, match="required"):
            to_andl(net, None)
        with pytest.raises(ValueError, match="covers 2"):
            to_andl(net, [1.0, 2.0])

    def test_unknown_marking_name_is_refused(self):
        net = parse_crs("Food: a, b\nr1 : a + b [c] => c\n")
        with pytest.raises(ValueError, match="unknown molecules"):
            to_andl(net, 1.0, marking={"nope": 3})

    def test_uncatalysable_reaction_omitted_and_counted_without_k_uncat(self):
        net = parse_crs("Food: a, b\nr1 : a + b [c] => c\nr2 : a [] => q\n")
        text = to_andl(net, 1.0)
        assert "1 reaction(s) omitted" in text
        assert "t_r2" not in text

    def test_uncatalysable_reaction_exports_background_with_k_uncat(self):
        net = parse_crs("Food: a, b\nr1 : a + b [c] => c\nr2 : a [] => q\n")
        text = to_andl(net, 1.0, k_uncat=0.01)
        assert "omitted" not in text
        assert "t_4_r2_u" in text


def test_write_andl_round_trips_through_a_file(tmp_path):
    net = parse_crs("Food: a, b\nr1 : a + b [c] => c\n")
    path = tmp_path / "net.andl"
    write_andl(net, 1.0, path)
    assert path.read_text().startswith("/*")


# --- Spike smoke tests: run only where the binary exists ---------------------------

def _spike():
    cand = os.environ.get("SPIKE_BIN") or shutil.which("spike") or str(
        Path.home() / "tools/spike/spike-1.6.0rc2-linux64/spike-1.6.0rc2-linux64")
    return cand if Path(cand).exists() else None


spike_bin = _spike()
needs_spike = pytest.mark.skipif(spike_bin is None,
                                 reason="Spike binary not found (set SPIKE_BIN)")


@needs_spike
def test_spike_parses_a_representative_export(tmp_path):
    net = parse_crs("Food: a, b\nr1 : a + b [c] => c\nr2 : c + c [] => a\n")
    path = tmp_path / "net.andl"
    write_andl(net, 1.0, path, k_uncat=0.05, food_influx=10.0, washout=0.01)
    out = subprocess.run([spike_bin, "load", f"-f={path}"], capture_output=True,
                         text=True, timeout=60)
    assert "ERROR" not in out.stdout + out.stderr
    # Positive control: the parser DOES report errors on a broken file.
    bad = tmp_path / "bad.andl"
    bad.write_text(path.read_text().replace("MassAction", ":::", 1))
    out = subprocess.run([spike_bin, "load", f"-f={bad}"], capture_output=True,
                         text=True, timeout=60)
    assert "ERROR" in out.stdout + out.stderr


@needs_spike
def test_spike_source_transition_fires_at_constant_rate(tmp_path):
    """The one semantic the three-toy calibration did not measure: an empty-preset
    transition's MassAction propensity is the bare constant. E[X](t) = rate * t."""
    net = parse_crs("Food: x\n")
    write_andl(net, [], tmp_path / "src.andl", marking={"x": 0}, food_influx=50.0)
    (tmp_path / "conf.spc").write_text("""
import: { from: "./src.andl"; }
configuration: {
  simulation: {
    name: "src";
    type: stochastic: { solver: direct: { threads: 4; runs: 2000; } }
    interval: 0:5:0.5;
    export: { places: []; csv: { sep: ";"; file: "./src.csv"; } }
  }
}
""")
    subprocess.run([spike_bin, "exe", "-f=conf.spc"], cwd=tmp_path,
                   capture_output=True, text=True, timeout=120)
    last = (tmp_path / "src.csv").read_text().strip().splitlines()[-1]
    t, x = (float(v) for v in last.split(";")[:2])
    assert t == pytest.approx(0.5)
    # E = 25, per-run sd = 5, 2000 runs -> SE ~ 0.11; 1.0 is ~9 SE yet catches any
    # misreading of the rate (a factor of 2 sits 220 SE away).
    assert x == pytest.approx(25.0, abs=1.0)
