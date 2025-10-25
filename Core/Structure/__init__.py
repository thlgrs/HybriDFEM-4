"""
HybriDFEM - Hybrid Finite Element / Discrete Element Method

A Python package for hybrid FEM-DEM structural analysis combining
deformable finite elements with rigid block assemblies.

Architecture:
    LOW-LEVEL (Monolithic API):
    - Structure_2D: Base class with node management
    - Structure_block: Rigid block structures
    - Structure_FEM: Finite element structures
    - Hybrid: Combined FEM-DEM structures
    
    HIGH-LEVEL (Builder API):
    - Build_blocks: Convenient block pattern generation
    - Build_FEM: Convenient mesh generation
    - Build_Hybrid: Combined hybrid builder

Author: HybriDFEM Team
Version: 4.0.0-alpha
Date: 2025
"""

# Version
__version__ = '4.0.0-alpha'
__author__ = 'HybriDFEM Team'

# Low-level API (monolithic)
from .structure_2d import Structure_2D
from .structure_block import Structure_block
from .structure_fem import Structure_FEM
from .structure_hybrid import Hybrid

# High-level API (builders)
from .build_blocks import Build_blocks
from .build_fem import Build_FEM
from .build_hybrid import Build_Hybrid

# Utilities
from . import utils

# Define public API
__all__ = [
    # Low-level classes
    'Structure_2D',
    'Structure_block',
    'Structure_FEM',
    'Hybrid',
    
    # High-level builders
    'Build_blocks',
    'Build_FEM',
    'Build_Hybrid',
    
    # Utilities
    'utils',
]


def print_architecture():
    """Print the architecture hierarchy."""
    print("""
HybriDFEM Architecture
======================

LOW-LEVEL API (Monolithic - Precise Control):
    Structure_2D (Abstract Base)
    ├── add_node([x, y]) → node_id
    ├── finalize()
    └── solve_linear()
        │
        ├─── Structure_block
        │    ├── add_rigid_block_by_nodes([n0,n1,n2,n3], ...)
        │    
        ├─── Structure_FEM
        │    ├── add_triangle_element([n0,n1,n2], E, nu, t)
        │    ├── add_beam_element([n0,n1], E, nu, H, b)
        │    
        └─── Hybrid (Multiple Inheritance)
             ├── Inherits all from Structure_block
             ├── Inherits all from Structure_FEM
             └── connect_node_to_block(node_id, block_id)

HIGH-LEVEL API (Builders - Convenience):
    Build_blocks (Inherits Structure_block)
    ├── add_wall(origin, L, H, pattern)
    ├── add_arch(center, span, rise, n_voussoirs)
    ├── add_voronoi_surface(region, n_cells)
    └── make_nodes()  # Converts geometry → nodes
    
    Build_FEM (Inherits Structure_FEM)
    ├── add_plate(origin, Lx, Ly, nx, ny)
    ├── add_mesh(region, element_size)
    └── make_nodes()  # Converts geometry → nodes
    
    Build_Hybrid (Multiple Inheritance)
    ├── Inherits all from Build_blocks
    ├── Inherits all from Build_FEM
    ├── Inherits all from Hybrid
    └── make_nodes()  # Converts both types

Usage Examples:
--------------
# Low-level (precise):
    hybrid = Hybrid()
    n0 = hybrid.add_node([0, 0])
    hybrid.add_triangle_element([n0, n1, n2], E=30e9, nu=0.2)
    hybrid.finalize()
    hybrid.solve_linear()

# High-level (convenient):
    bh = Build_Hybrid()
    bh.add_wall([0, 0], L=3, H=2)
    bh.add_plate([0, 2], Lx=3, Ly=0.5, nx=6, ny=1)
    bh.make_nodes()
    bh.solve_linear()

# Mixed (power users):
    bh = Build_Hybrid()
    bh.add_wall([0, 0], L=3, H=2)
    bh.make_nodes()
    # Now use low-level methods for custom additions
    n_extra = bh.add_node([3.5, 1.0])
    bh.solve_linear()
    """)


def get_version():
    """Get version string."""
    return __version__


# Print info on import (can be disabled)
_PRINT_ON_IMPORT = False
if _PRINT_ON_IMPORT:
    print(f"HybriDFEM v{__version__} loaded")
    print("Use hybridfem.print_architecture() to see class hierarchy")
