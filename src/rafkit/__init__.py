"""rafkit -- autocatalytic (RAF) sets in catalytic reaction networks.

A small, dependency-light implementation of RAF theory (Hordijk & Steel 2004):
maximal RAFs, the self-referential ("strictly autocatalytic") variant, irreducible
RAF sampling, Kauffman's binary polymer model as a generator, and stochastic simulation
of a network so that subRAFs can be watched seeding themselves into existence.

Every algorithm here is written from the published papers and carries hand-computed
known-answer tests, because a RAF algorithm that is subtly wrong produces plausible
numbers rather than errors.

See the README for the calibration against Steel, Hordijk & Smith (2012) and for
CatReNet interoperability.
"""
from rafkit.binary_polymer import BinaryPolymerNetwork, binary_polymer
from rafkit.catalysis import catalysing_molecules, is_catalysed, normalise
from rafkit.crs import parse_crs, read_crs, to_crs, write_crs
from rafkit.gillespie import Trajectory, propensities, simulate
from rafkit.inhibition import (classes_from_inhibitors, is_uninhibited, is_uraf,
                               max_urafs, support)
from rafkit.network import ReactionNetwork
from rafkit.pnml import to_pnml, write_pnml
from rafkit.raf import (
    RafResult, catrenet_strictly_autocatalytic, core_raf, exploitability,
    has_unique_irraf, irrraf_census, is_food_catalysed, max_raf, max_raf_strict,
    sample_irrraf,
)

__version__ = "0.4.0"

__all__ = [
    "BinaryPolymerNetwork", "binary_polymer", "ReactionNetwork",
    "RafResult", "max_raf", "max_raf_strict", "sample_irrraf", "irrraf_census",
    "exploitability", "is_food_catalysed", "catrenet_strictly_autocatalytic",
    "core_raf", "has_unique_irraf",
    "is_catalysed", "catalysing_molecules", "normalise",
    "parse_crs", "read_crs", "to_crs", "write_crs",
    "to_pnml", "write_pnml",
    "simulate", "propensities", "Trajectory",
    "max_urafs", "is_uraf", "is_uninhibited", "support", "classes_from_inhibitors",
]
