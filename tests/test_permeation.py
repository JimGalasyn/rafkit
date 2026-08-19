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
R = 0.5
V_COMP = (4.0 / 3.0) * np.pi * R ** 3      # 0.5236
A_COMP = 4.0 * np.pi * R ** 2              # 3.1416
V_VOX = 2.5 ** 2                           # 6.25
GEOM = dict(permeability=1.0, area=A_COMP, volume_in=V_COMP, volume_out=V_VOX)


class TestTheVolumeTrap:
    """A concentration difference is not a count difference unless the volumes match."""

    def test_the_two_volumes_really_do_differ(self):
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
        """Exchange rate is permeability*area/volume, so halving the compartment volume at
        fixed concentration difference doubles the fractional response."""
        n_in, n_out = np.array([0.0]), np.array([100.0])
        big = permeation_flux(n_in, n_out, **{**GEOM, 'volume_in': 2 * V_COMP})
        small = permeation_flux(n_in, n_out, **GEOM)
        assert big[0] == pytest.approx(small[0])          # influx into an empty compartment
        # but as a fraction of what the compartment can hold at that concentration, small is 2x
        assert small[0] / V_COMP == pytest.approx(2 * big[0] / (2 * V_COMP))


class TestConservationAndBounds:
    def test_flux_is_a_single_quantity_moved_between_two_places(self):
        """Whatever enters the compartment must leave the medium: one number, applied twice."""
        n_in, n_out = np.array([3.0, 40.0]), np.array([80.0, 5.0])
        f = permeation_flux(n_in, n_out, **GEOM)
        assert (n_in + f).sum() + (n_out - f).sum() == pytest.approx(n_in.sum() + n_out.sum())

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
