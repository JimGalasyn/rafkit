"""Chemistry observables and network generators for the abiogenesis layer.

See `docs/DESIGN_abiogenesis.md` §10 for the intended layout. This package is the
observables-and-statistics half; the chemistry itself lives in
`morphospace/physics/protocell/`.
"""
from morphospace.chemistry.binary_polymer import (
    BinaryPolymerNetwork, binary_polymer,
)
from morphospace.chemistry.raf import RafResult, exploitability, max_raf

__all__ = ["BinaryPolymerNetwork", "binary_polymer", "RafResult", "max_raf",
           "exploitability"]
