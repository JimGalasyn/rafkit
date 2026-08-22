"""rafkit -- autocatalytic (RAF) sets in catalytic reaction networks.

A small, dependency-light implementation of RAF theory (Hordijk & Steel 2004):
maximal RAFs, the self-referential ("strictly autocatalytic") variant, irreducible
RAF sampling, Kauffman's binary polymer model as a generator, and stochastic simulation
of a network so that subRAFs can be watched seeding themselves into existence.

Four modules are deliberately off that theme. `dilution` reproduces a two-species
autocatalytic ODE with no RAF structure at all, because it is the library's only
*analytic* calibration target. `permeation` is one line of membrane transport physics,
there because it is a modelling choice that changes the answer and is easy to get
wrong in a way that produces plausible numbers. `autocatalysis` decides whether a set of
reactions can all run forward at once, which is a thermodynamic question RAF theory cannot pose
and is this project's external calibration anchor. `thermo` supplies the free energy the
rest of the library leaves implicit -- unit rate constants in both directions assert
that every polymer is isoenergetic with its parts -- and makes catalysis a ratio
applied to both directions rather than a licence to exist. See their docstrings.

Every algorithm here is written from the published papers and carries hand-computed
known-answer tests, because a RAF algorithm that is subtly wrong produces plausible
numbers rather than errors.

See the README for the calibration against Steel, Hordijk & Smith (2012) and for
CatReNet interoperability.
"""
from rafkit.autocatalysis import (Consistency, affinities,
                                  is_thermodynamically_consistent, stoichiometry)
from rafkit.binary_polymer import BinaryPolymerNetwork, binary_polymer
from rafkit.complementary_polymer import (complement,
                                          complementary_polymer)
from rafkit.firing_disk import firing_disk_polymer
from rafkit.catalysis import catalysing_molecules, is_catalysed, normalise
from rafkit.crs import parse_crs, read_crs, to_crs, write_crs
from rafkit.dilution import (DilutionResult, flux_linear, flux_quadratic,
                             is_bistable, run_cstr, run_serial_dilution,
                             symmetric_growth_ratio)
from rafkit.gillespie import Trajectory, propensities, simulate
from rafkit.inhibition import (classes_from_inhibitors, is_uninhibited, is_uraf,
                               max_urafs, support)
from rafkit.network import ReactionNetwork
from rafkit.permeation import permeable_by_length, permeation_flux
from rafkit.pnml import to_pnml, write_pnml
from rafkit.thermo import (RT_KCAL_PER_MOL_298K, RT_KJ_PER_MOL_298K, BondEnergies,
                           Kinetics, Rates, barrier_drop_from_enhancement,
                           catalysed_rates, kinetics_from_energies, unpaired_catalysis,
                           detailed_balance_residual, elongation_ratio,
                           enhancement_from_barrier_drop, equilibrium_constant,
                           mean_length, rate_constants, reaction_free_energies,
                           reaction_rate_constants, reversible_pairs,
                           sequence_correlation_length, transfer_matrix)
from rafkit.raf import (
    RafResult, catrenet_strictly_autocatalytic, core_raf, exploitability,
    has_unique_irraf, irrraf_census, is_food_catalysed, max_raf, max_raf_strict,
    sample_irrraf,
)

__version__ = "0.6.0"

__all__ = [
    "BinaryPolymerNetwork", "binary_polymer",
    "Consistency", "is_thermodynamically_consistent", "stoichiometry", "affinities",
    "complementary_polymer",
    "firing_disk_polymer",
    "complement", "ReactionNetwork",
    "RafResult", "max_raf", "max_raf_strict", "sample_irrraf", "irrraf_census",
    "exploitability", "is_food_catalysed", "catrenet_strictly_autocatalytic",
    "core_raf", "has_unique_irraf",
    "is_catalysed", "catalysing_molecules", "normalise",
    "parse_crs", "read_crs", "to_crs", "write_crs",
    "DilutionResult", "run_serial_dilution", "run_cstr", "is_bistable",
    "flux_linear", "flux_quadratic", "symmetric_growth_ratio",
    "permeation_flux", "permeable_by_length",
    "BondEnergies", "Rates", "RT_KJ_PER_MOL_298K", "RT_KCAL_PER_MOL_298K",
    "equilibrium_constant", "rate_constants", "catalysed_rates",
    "enhancement_from_barrier_drop", "barrier_drop_from_enhancement",
    "reaction_free_energies", "reaction_rate_constants", "reversible_pairs",
    "detailed_balance_residual", "transfer_matrix", "elongation_ratio", "mean_length",
    "sequence_correlation_length", "unpaired_catalysis",
    "Kinetics", "kinetics_from_energies",
    "to_pnml", "write_pnml",
    "simulate", "propensities", "Trajectory",
    "max_urafs", "is_uraf", "is_uninhibited", "support", "classes_from_inhibitors",
]
