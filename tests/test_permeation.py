"""Size-selective membrane permeation -- the volume trap, and the properties that catch it.

These are deliberately DETERMINISTIC checks. The result that motivated this module is a
stochastic reproduction of a published figure with ~200 s of simulation and a standard
deviation of +-6 on a mean of 16; that belongs in the downstream research client as a
programme gate, not in a library test suite, where a gate should be fast and exact.

What is testable here exactly is that the flux obeys Fick's law across a membrane, and in
particular that it responds to a CONCENTRATION difference rather than a COUNT difference --
the distinction that cost a published reproduction until it was found.
"""
from __future__ import annotations

import numpy as np
import pytest

from rafkit.permeation import permeable_by_length, permeation_flux

# The geometry of Hordijk et al. (2018): a compartment of radius 0.5 in a 2.5 x 2.5 voxel.
# ⚠ Both sides must be VOLUMES. Their world is two-dimensional -- "if the world type is set to
# 2D then Y is forced to 1" -- so a voxel is a slab of unit thickness, and `SLAB` makes that
# explicit rather than leaving a voxel AREA to be compared against a sphere VOLUME.
R = 0.5
SLAB = 1.0                                 # the 2D world's thickness, from "Y is forced to 1"
V_COMP = (4.0 / 3.0) * np.pi * R ** 3      # 0.5236, a sphere
A_COMP = 4.0 * np.pi * R ** 2              # 3.1416, its surface
V_VOX = 2.5 * 2.5 * SLAB                   # 6.25, a slab
GEOM = dict(permeability=1.0, area=A_COMP, volume_in=V_COMP, volume_out=V_VOX)


class TestTheVolumeTrap:
    """A concentration difference is not a count difference unless the volumes match."""

    def test_the_two_volumes_really_do_differ(self):
        """The module's headline number, and it must be a ratio of two volumes."""
        assert V_VOX / V_COMP == pytest.approx(11.9, rel=0.02)

    def test_equal_CONCENTRATIONS_give_zero_flux(self):
        n_in = np.array([10.0])
        n_out = n_in / V_COMP * V_VOX                 # same concentration, 11.9x the count
        assert permeation_flux(n_in, n_out, **GEOM) == pytest.approx([0.0], abs=1e-12)

    def test_equal_COUNTS_give_NON_zero_flux(self):
        """The trap stated directly: with unequal volumes, equal counts are NOT equilibrium.
        A count-difference implementation returns zero here and is wrong."""
        n = np.array([10.0])
        flux = permeation_flux(n, n.copy(), **GEOM)
        assert abs(flux[0]) > 1.0
        assert flux[0] < 0        # the compartment is denser, so the net flux is OUTWARD

    def test_direction_follows_concentration_not_count(self):
        """The compartment holds FEWER molecules but at HIGHER concentration, so it exports."""
        n_in, n_out = np.array([5.0]), np.array([20.0])
        assert n_in[0] < n_out[0]
        assert n_in[0] / V_COMP > n_out[0] / V_VOX
        assert permeation_flux(n_in, n_out, **GEOM)[0] < 0

    def test_smaller_compartment_equilibrates_faster(self):
        """Exchange rate is permeability*area/volume, so at the same CONTENTS a smaller
        compartment changes concentration faster.

        ⚠ This must be tested with a NON-EMPTY compartment. An earlier version used n_in = 0,
        which zeroes the `n_in / volume_in` term outright -- so the flux no longer depended on
        volume_in at all and the two arms were equal by construction. The test asserted that
        equality and passed, proving nothing about the property it named.
        """
        n_in, n_out = np.array([5.0]), np.array([100.0])
        small = permeation_flux(n_in, n_out, **GEOM)
        big = permeation_flux(n_in, n_out, **{**GEOM, 'volume_in': 2 * V_COMP})
        assert small[0] != pytest.approx(big[0])       # the volume MUST matter here
        # A denser small compartment takes in fewer molecules ...
        assert small[0] < big[0]
        # ... but its concentration moves further, which is what "equilibrates faster" means.
        assert small[0] / V_COMP > big[0] / (2 * V_COMP)


class TestConservationAndBounds:
    def test_flux_equals_the_analytic_fick_value(self):
        """⚠ This replaces a test asserting `(n_in+f).sum() + (n_out-f).sum()` equals the
        starting total. That is algebraically true for ANY f -- the f cancels -- so it was
        vacuous, and gave false confidence in exactly the invariant it named. What is
        actually checkable is the value, against an independently written expression."""
        n_in, n_out = np.array([3.0, 40.0]), np.array([80.0, 5.0])
        f = permeation_flux(n_in, n_out, dt=0.01, **GEOM)
        expected = GEOM['permeability'] * A_COMP * 0.01 * (n_out / V_VOX - n_in / V_COMP)
        assert f == pytest.approx(expected)
        assert f[0] > 0 and f[1] < 0          # and the two species move in OPPOSITE directions

    def test_repeated_application_converges_to_equal_CONCENTRATIONS(self):
        """The substantive conservation statement: iterating the flux at a stable step drives
        the two sides to equal concentration -- not equal counts -- and conserves the total."""
        n_in, n_out = np.array([0.0]), np.array([100.0])
        total = n_in.sum() + n_out.sum()
        # Relaxation rate is P*A*(1/V_in + 1/V_out) ~ 6.5, so 6000 steps of dt=1e-3 is ~39
        # e-foldings: fully converged, and the tolerance can be tight rather than fitted to
        # however far a shorter run happened to get.
        for _ in range(6000):
            f = permeation_flux(n_in, n_out, dt=0.001, **GEOM)
            n_in, n_out = n_in + f, n_out - f
        assert n_in.sum() + n_out.sum() == pytest.approx(total)
        assert n_in[0] / V_COMP == pytest.approx(n_out[0] / V_VOX, rel=1e-9)
        assert n_in[0] != pytest.approx(n_out[0])     # equal CONCENTRATION, unequal counts

    def test_cannot_remove_more_than_the_compartment_holds(self):
        n_in, n_out = np.array([2.0]), np.array([0.0])
        f = permeation_flux(n_in, n_out, **{**GEOM, 'permeability': 1e6})
        assert f[0] >= -n_in[0]
        assert (n_in + f)[0] >= 0.0

    def test_cannot_remove_more_than_the_medium_holds(self):
        n_in, n_out = np.array([0.0]), np.array([2.0])
        f = permeation_flux(n_in, n_out, **{**GEOM, 'permeability': 1e6})
        assert f[0] <= n_out[0]
        assert (n_out - f)[0] >= 0.0

    def test_zero_permeability_moves_nothing(self):
        f = permeation_flux(np.array([5.0]), np.array([90.0]), **{**GEOM, 'permeability': 0.0})
        assert f == pytest.approx([0.0])

    def test_volumes_must_be_positive(self):
        """There is no default volume, because a default is exactly how counts get passed
        where concentrations are meant."""
        with pytest.raises(ValueError, match="volumes must be positive"):
            permeation_flux(np.array([1.0]), np.array([1.0]), permeability=1.0, area=1.0,
                            volume_in=0.0, volume_out=1.0)

    def test_flux_scales_linearly_with_dt(self):
        a = permeation_flux(np.array([1.0]), np.array([90.0]), dt=0.01, **GEOM)
        b = permeation_flux(np.array([1.0]), np.array([90.0]), dt=0.02, **GEOM)
        assert b[0] == pytest.approx(2 * a[0])


class TestInputHandling:
    def test_scalar_inputs_return_an_indexable_array(self):
        """A 0-d result makes `result[0]` raise IndexError, which is a nuisance in exactly the
        one-species case a caller reaches for first."""
        f = permeation_flux(5.0, 100.0, permeability=1.0, area=1.0,
                            volume_in=1.0, volume_out=1.0)
        assert f.shape == (1,)
        assert f[0] == pytest.approx(95.0)

    def test_negative_permeability_or_area_is_refused(self):
        """A negative coefficient silently reverses the flux -- uphill transport, reported as
        a plausible number."""
        for bad in (dict(permeability=-1.0), dict(area=-1.0)):
            with pytest.raises(ValueError, match="non-negative"):
                permeation_flux(np.array([1.0]), np.array([1.0]),
                                **{**dict(permeability=1.0, area=1.0,
                                          volume_in=1.0, volume_out=1.0), **bad})

    def test_a_bare_species_string_is_refused(self):
        """`permeable_by_length("011", 2)` would iterate to ('0','1','1') and return a
        per-CHARACTER mask -- a wrong answer rather than an error."""
        with pytest.raises(TypeError, match="not a single string"):
            permeable_by_length("011", max_len=2)

    def test_integer_lengths_are_accepted_directly(self):
        assert list(permeable_by_length([1, 2, 3, 2], max_len=2)) == [True, True, False, True]


class TestSizeSelectivity:
    def test_impermeable_species_do_not_move(self):
        n_in, n_out = np.array([1.0, 1.0]), np.array([90.0, 90.0])
        f = permeation_flux(n_in, n_out, permeable=np.array([True, False]), **GEOM)
        assert f[0] > 0
        assert f[1] == 0.0

    def test_aperture_admits_short_species_only(self):
        species = ("0", "11", "011", "110", "01", "1")
        mask = permeable_by_length(species, max_len=2)
        assert list(mask) == [True, True, False, False, True, True]

    def test_blocking_isolates_one_species_regardless_of_size(self):
        """How a treatment arm isolates a single molecule's transport -- the manipulation the
        induction experiment turns on, where the two arms differ in exactly this one bit."""
        species = ("0", "11", "011", "110", "01", "1")
        mask = permeable_by_length(species, max_len=2, blocked=(4,))
        assert mask[4] == np.False_          # 01 held back although it is short enough
        assert list(mask) == [True, True, False, False, False, True]

    def test_blocking_does_not_mutate_the_unblocked_mask(self):
        species = ("0", "11", "01")
        a = permeable_by_length(species, max_len=2)
        b = permeable_by_length(species, max_len=2, blocked=(2,))
        assert list(a) == [True, True, True]
        assert list(b) == [True, True, False]


class TestBatchedGeometry:
    """A population of compartments, each with its own size, stepped at once.

    The case a spatial model needs: every pore holds a cell, and the cells differ in radius.
    Looping `permeation_flux` per compartment works but puts a Python call in the inner loop
    of an integrator; broadcasting the geometry keeps ONE implementation of the law instead
    of a fast inline copy that could drift from it.

    ⚠ These tests are parameterised over WHICH operand carries the batch. The first version
    put the batch on `volume_in` every time -- and the implementation happened to key its
    reshape off `volume_in`, so the suite exercised the one shape that worked and shipped a
    silent wrong answer for the others. A batched `area` with a scalar `volume_in` broadcast
    along the SPECIES axis, returning [1.6, 1.6, 1.6, 1.6] where [1.6, 3.2, 4.8, 6.4] was
    correct. Testing only the path the code takes is not testing the contract.
    """

    BASE = dict(permeability=1.0, area=A_COMP, volume_in=V_COMP, volume_out=V_VOX)

    @pytest.mark.parametrize("operand", ["permeability", "area", "volume_in", "volume_out"])
    def test_every_geometry_operand_can_carry_the_batch(self, operand):
        """Uniform values in a batched operand must reproduce the scalar answer exactly,
        whichever operand is batched."""
        n_in, n_out = np.array([3.0, 40.0]), np.array([80.0, 5.0])
        one = permeation_flux(n_in, n_out, **self.BASE)
        kw = dict(self.BASE)
        kw[operand] = np.full(4, self.BASE[operand])
        many = permeation_flux(np.tile(n_in, (4, 1)), np.tile(n_out, (4, 1)), **kw)
        assert many.shape == (4, 2)
        for row in many:
            assert row == pytest.approx(one)

    @pytest.mark.parametrize("operand", ["permeability", "area"])
    def test_a_batched_multiplier_scales_each_compartment(self, operand):
        """The test that catches the species-axis bug: with n_species == n_compartments the
        wrong broadcast is silent, so the values must be checked, not just the shape."""
        n_in = np.zeros((4, 4))
        n_out = np.full((4, 4), 10.0)
        kw = dict(permeability=1.0, area=1.0, volume_in=1.0, volume_out=6.25)
        kw[operand] = np.array([1.0, 2.0, 3.0, 4.0])
        f = permeation_flux(n_in, n_out, **kw)
        expected = np.array([1.0, 2.0, 3.0, 4.0]) * (10.0 / 6.25)
        assert f[:, 0] == pytest.approx(expected)
        assert (f[:, 0] == f[0, 0]).sum() == 1, "flux must vary by compartment, not be uniform"

    def test_batched_volume_out_scales_each_compartment(self):
        f = permeation_flux(np.zeros((2, 3)), np.full((2, 3), 10.0), permeability=1.0,
                            area=1.0, volume_in=1.0, volume_out=np.array([6.25, 12.5]))
        assert f[0, 0] == pytest.approx(2 * f[1, 0])

    def test_compartments_of_different_size_get_different_flux(self):
        """If the per-compartment volumes were not really being used, the rows would agree."""
        n_in = np.tile(np.array([5.0, 5.0]), (3, 1))
        n_out = np.tile(np.array([90.0, 90.0]), (3, 1))
        vin = np.array([V_COMP, 2 * V_COMP, 4 * V_COMP])
        f = permeation_flux(n_in, n_out, permeability=1.0, area=A_COMP,
                            volume_in=vin, volume_out=V_VOX)
        assert f[0, 0] != pytest.approx(f[1, 0])
        # a denser (smaller) compartment has a smaller inward concentration gradient
        assert f[0, 0] < f[1, 0] < f[2, 0]

    def test_batched_still_clips_per_compartment(self):
        n_in = np.array([[2.0], [100.0]])
        n_out = np.array([[0.0], [0.0]])
        f = permeation_flux(n_in, n_out, permeability=1e6, area=A_COMP,
                            volume_in=np.array([V_COMP, V_COMP]), volume_out=V_VOX)
        assert (n_in + f >= 0).all()
        assert f[0, 0] == pytest.approx(-2.0)
        assert f[1, 0] == pytest.approx(-100.0)

    def test_batched_volumes_are_still_validated(self):
        with pytest.raises(ValueError, match="volumes must be positive"):
            permeation_flux(np.zeros((2, 1)), np.zeros((2, 1)), permeability=1.0, area=1.0,
                            volume_in=np.array([1.0, 0.0]), volume_out=1.0)

    @pytest.mark.parametrize("operand", ["permeability", "area", "volume_in", "volume_out"])
    def test_a_per_species_vector_is_refused_not_broadcast(self, operand):
        """Geometry is per-COMPARTMENT. A per-species vector cannot be distinguished from a
        per-compartment one when the counts coincide, so guessing is what produced the silent
        wrong answer. It must raise, and say where per-species variation belongs."""
        kw = dict(permeability=1.0, area=1.0, volume_in=1.0, volume_out=6.25)
        kw[operand] = np.array([1.0, 2.0, 3.0])          # 3 species, 4 compartments
        with pytest.raises(ValueError, match="per-compartment"):
            permeation_flux(np.zeros((4, 3)), np.ones((4, 3)), **kw)

    def test_per_species_variation_goes_through_permeable(self):
        """The escape hatch the error message points at: `permeable` multiplies the flux, so
        it carries graded per-species factors and not only a 0/1 mask."""
        n_in, n_out = np.zeros((2, 3)), np.full((2, 3), 10.0)
        full = permeation_flux(n_in, n_out, permeability=1.0, area=1.0,
                               volume_in=1.0, volume_out=6.25)
        graded = permeation_flux(n_in, n_out, permeability=1.0, area=1.0, volume_in=1.0,
                                 volume_out=6.25, permeable=np.array([1.0, 0.5, 0.0]))
        assert graded[0] == pytest.approx(full[0] * np.array([1.0, 0.5, 0.0]))
