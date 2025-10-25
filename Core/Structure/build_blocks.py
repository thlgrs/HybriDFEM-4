"""
Build_blocks - High-level block structure builder

This module provides convenience methods for creating common block patterns
(walls, arches, Voronoi tessellations, etc.).

Author: HybriDFEM Team
Date: 2025
"""

import numpy as np
import warnings
from typing import List, Optional, Tuple
from structure_block import Structure_block


class Build_blocks(Structure_block):
    """
    High-level block structure builder.
    
    For users who want convenience methods for common block patterns.
    Stores geometry, then converts to nodes via make_nodes().
    
    Workflow:
        1. Add geometry: add_wall(), add_arch(), etc.
        2. Convert: make_nodes()
        3. Solve: solve_linear()
    
    Usage:
        bb = Build_blocks()
        bb.add_wall([0, 0], L=3.0, H=2.0, pattern='running_bond')
        bb.add_arch([1.5, 2.0], span=2.0, rise=1.0, n_voussoirs=9)
        bb.make_nodes()
        bb.solve_linear()
    """
    
    def __init__(self):
        """Initialize builder."""
        super().__init__()
        
        # High-level geometry storage (before conversion)
        self._block_geometry = []
        self._nodes_made = False
    
    # =========================================================================
    # HIGH-LEVEL GEOMETRY API (Stores geometry)
    # =========================================================================
    
    def add_wall(self, origin: np.ndarray, L: float, H: float,
                 pattern: str = 'running_bond',
                 rho: float = 2400, b: float = 0.2,
                 material=None):
        """
        Add masonry wall (stores geometry for later conversion).
        
        Parameters:
            origin: [x, y] bottom-left corner
            L: Wall length (m)
            H: Wall height (m)
            pattern: Bonding pattern:
                - 'running_bond': Standard offset pattern
                - 'stack_bond': Aligned vertically
                - 'flemish_bond': Alternating headers/stretchers
            rho: Density (kg/m³)
            b: Thickness (m)
            material: Material object (optional)
        
        Example:
            bb = Build_blocks()
            bb.add_wall([0, 0], L=3.0, H=2.0, pattern='running_bond')
            bb.make_nodes()
        
        Note:
            Uses standard block dimensions (0.4m x 0.2m).
            Call make_nodes() to convert to actual nodes and blocks.
        """
        if self._nodes_made:
            raise RuntimeError(
                "Cannot add geometry after make_nodes(). "
                "Create a new builder or use add_node() directly."
            )
        
        self._block_geometry.append({
            'type': 'wall',
            'origin': np.array(origin, dtype=float),
            'L': L,
            'H': H,
            'pattern': pattern,
            'rho': rho,
            'b': b,
            'material': material
        })
        
        print(f"Added {pattern} wall at {origin} (L={L}, H={H})")
    
    def add_beam(self, N1: np.ndarray, N2: np.ndarray, 
                 n_blocks: int, H: float,
                 rho: float = 2400, b: float = 0.2):
        """
        Add horizontal beam of blocks.
        
        Parameters:
            N1, N2: Start and end points [x, y]
            n_blocks: Number of blocks along beam
            H: Block height (m)
            rho: Density (kg/m³)
            b: Thickness (m)
        
        Example:
            bb.add_beam([0, 2], [3, 2], n_blocks=6, H=0.2)
        """
        if self._nodes_made:
            raise RuntimeError("Cannot add geometry after make_nodes()")
        
        self._block_geometry.append({
            'type': 'beam',
            'N1': np.array(N1, dtype=float),
            'N2': np.array(N2, dtype=float),
            'n_blocks': n_blocks,
            'H': H,
            'rho': rho,
            'b': b
        })
        
        print(f"Added beam from {N1} to {N2} ({n_blocks} blocks)")
    
    def add_arch(self, center: np.ndarray, span: float, rise: float,
                 n_voussoirs: int, thickness: float,
                 rho: float = 2400, b: float = 0.2):
        """
        Add masonry arch.
        
        Parameters:
            center: [x, y] center of arch base
            span: Horizontal span (m)
            rise: Vertical rise (m)
            n_voussoirs: Number of voussoir blocks
            thickness: Voussoir radial thickness (m)
            rho: Density (kg/m³)
            b: Depth (out-of-plane, m)
        
        Example:
            bb.add_arch([1.5, 2.0], span=2.0, rise=1.0, 
                       n_voussoirs=9, thickness=0.3)
        
        Note:
            Creates circular arch. For pointed arch, use add_pointed_arch().
        """
        if self._nodes_made:
            raise RuntimeError("Cannot add geometry after make_nodes()")
        
        self._block_geometry.append({
            'type': 'arch',
            'center': np.array(center, dtype=float),
            'span': span,
            'rise': rise,
            'n_voussoirs': n_voussoirs,
            'thickness': thickness,
            'rho': rho,
            'b': b
        })
        
        print(f"Added arch at {center} (span={span}, rise={rise})")
    
    def add_voronoi_surface(self, region_corners: List[List[float]], 
                           n_cells: int, thickness: float,
                           rho: float = 2400, b: float = 0.2,
                           seed: Optional[int] = None):
        """
        Add Voronoi tessellation surface.
        
        Parameters:
            region_corners: [[x1,y1], [x2,y2], ...] polygon vertices
            n_cells: Approximate number of Voronoi cells
            thickness: Block thickness (m)
            rho: Density (kg/m³)
            b: Depth (m)
            seed: Random seed for reproducibility
        
        Example:
            region = [[0, 0], [3, 0], [3, 2], [0, 2]]
            bb.add_voronoi_surface(region, n_cells=20, thickness=0.2, seed=42)
        
        Note:
            Requires scipy. Random point distribution unless seed specified.
        """
        if self._nodes_made:
            raise RuntimeError("Cannot add geometry after make_nodes()")
        
        self._block_geometry.append({
            'type': 'voronoi',
            'region': region_corners,
            'n_cells': n_cells,
            'thickness': thickness,
            'rho': rho,
            'b': b,
            'seed': seed
        })
        
        print(f"Added Voronoi tessellation ({n_cells} cells)")
    
    # =========================================================================
    # CONVERSION: Geometry → Nodes
    # =========================================================================
    
    def make_nodes(self):
        """
        Convert stored geometry to nodes and blocks.
        
        This is the KEY method that bridges high-level geometry
        and low-level monolithic API.
        
        Process:
        1. Convert each stored geometry to nodes (using add_node())
        2. Create blocks (using add_rigid_block_by_nodes())
        3. Detect and merge duplicate nodes
        4. Call finalize() to prepare for solving
        
        Example:
            bb = Build_blocks()
            bb.add_wall([0, 0], L=3, H=2)
            bb.add_arch([1.5, 2], span=2, rise=1, n_voussoirs=9)
            bb.make_nodes()  # ← Converts everything
            bb.solve_linear()
        """
        if self._nodes_made:
            warnings.warn("make_nodes() already called", RuntimeWarning)
            return
        
        print(f"\n{'='*60}")
        print(f"CONVERTING GEOMETRY TO NODES")
        print(f"{'='*60}")
        print(f"Stored geometries: {len(self._block_geometry)}")
        
        for i, geom in enumerate(self._block_geometry):
            geom_type = geom['type']
            print(f"\n[{i+1}/{len(self._block_geometry)}] Converting {geom_type}...")
            
            if geom_type == 'wall':
                self._build_wall(geom)
            elif geom_type == 'beam':
                self._build_beam(geom)
            elif geom_type == 'arch':
                self._build_arch(geom)
            elif geom_type == 'voronoi':
                self._build_voronoi(geom)
            else:
                raise ValueError(f"Unknown geometry type: {geom_type}")
        
        # Clear stored geometry (no longer needed)
        self._block_geometry.clear()
        self._nodes_made = True
        
        # Finalize structure
        print(f"\n{'='*60}")
        print(f"GEOMETRY CONVERSION COMPLETE")
        print(f"{'='*60}")
        print(f"Total nodes: {len(self._nodes)}")
        print(f"Total blocks: {len(self.list_blocks)}")
        
        self.finalize()
    
    # =========================================================================
    # INTERNAL: Geometry → Node Conversion
    # =========================================================================
    
    def _build_wall(self, geom: dict):
        """
        Convert wall geometry to nodes and blocks.
        
        Uses low-level methods (add_node, add_rigid_block_by_nodes)
        inherited from Structure_block.
        
        TODO: Implement different bonding patterns
        """
        origin = geom['origin']
        L, H = geom['L'], geom['H']
        pattern = geom['pattern']
        rho, b = geom['rho'], geom['b']
        
        if pattern == 'running_bond':
            self._build_running_bond_wall(origin, L, H, rho, b)
        elif pattern == 'stack_bond':
            self._build_stack_bond_wall(origin, L, H, rho, b)
        elif pattern == 'flemish_bond':
            self._build_flemish_bond_wall(origin, L, H, rho, b)
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
    
    def _build_running_bond_wall(self, origin: np.ndarray, L: float, H: float,
                                 rho: float, b: float):
        """
        Create running bond masonry pattern.
        
        Standard pattern:
        ┌─────┬─────┬─────┐
        │     │     │     │  Course 2 (offset)
        ├──┬──┴──┬──┴──┬──┤
        │  │     │     │  │  Course 1
        └──┴─────┴─────┴──┘
        
        TODO: Implement actual running bond creation
        """
        print("    Creating running bond pattern...")
        
        # Standard block dimensions
        block_L = 0.4  # Block length (m)
        block_H = 0.2  # Block height (m)
        
        # Compute number of courses and blocks per course
        n_courses = int(np.ceil(H / block_H))
        n_blocks_per_course = int(np.ceil(L / block_L))
        
        blocks_created = 0
        
        # TODO: Implement actual block creation
        # Pseudocode:
        # for course in range(n_courses):
        #     y_base = origin[1] + course * block_H
        #     x_offset = (block_L / 2) if (course % 2 == 1) else 0
        #     
        #     for i in range(n_blocks_per_course):
        #         # Create block vertices
        #         # Find or create nodes
        #         # Call add_rigid_block_by_nodes()
        
        print(f"    Created {blocks_created} blocks")
    
    def _build_stack_bond_wall(self, origin: np.ndarray, L: float, H: float,
                               rho: float, b: float):
        """Stack bond: blocks directly above each other."""
        print("    Creating stack bond pattern...")
        # TODO: Implement
        pass
    
    def _build_flemish_bond_wall(self, origin: np.ndarray, L: float, H: float,
                                 rho: float, b: float):
        """Flemish bond: alternating headers and stretchers."""
        print("    Creating Flemish bond pattern...")
        # TODO: Implement
        pass
    
    def _build_beam(self, geom: dict):
        """
        Convert beam geometry to blocks.
        
        TODO: Implement beam block creation
        """
        N1, N2 = geom['N1'], geom['N2']
        n_blocks = geom['n_blocks']
        H = geom['H']
        rho, b = geom['rho'], geom['b']
        
        print(f"    Creating {n_blocks} blocks along beam...")
        
        # TODO: Implement
        # Direction vector
        # vec = N2 - N1
        # L_total = np.linalg.norm(vec)
        # direction = vec / L_total
        # block_L = L_total / n_blocks
        # ...
        
        pass
    
    def _build_arch(self, geom: dict):
        """
        Convert arch geometry to voussoir blocks.
        
        TODO: Implement circular arch creation
        """
        center = geom['center']
        span = geom['span']
        rise = geom['rise']
        n_voussoirs = geom['n_voussoirs']
        thickness = geom['thickness']
        rho, b = geom['rho'], geom['b']
        
        print(f"    Creating arch with {n_voussoirs} voussoirs...")
        
        # TODO: Implement
        # Compute arch radius
        # radius = (rise**2 + (span/2)**2) / (2 * rise)
        # Compute angular span
        # For each voussoir, create trapezoidal block
        
        pass
    
    def _build_voronoi(self, geom: dict):
        """
        Convert Voronoi tessellation to blocks.
        
        TODO: Implement Voronoi tessellation
        Requires: scipy.spatial.Voronoi
        """
        region = np.array(geom['region'])
        n_cells = geom['n_cells']
        rho, b = geom['rho'], geom['b']
        seed = geom['seed']
        
        print(f"    Creating Voronoi tessellation...")
        
        # TODO: Implement
        # from scipy.spatial import Voronoi
        # Generate random points in region
        # Compute Voronoi diagram
        # For each cell, create block
        
        pass
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def _find_or_create_node(self, position: np.ndarray, 
                            tolerance: float = 1e-6) -> int:
        """
        Find existing node at position or create new one.
        
        This prevents duplicate nodes at block interfaces.
        
        Parameters:
            position: [x, y] coordinates
            tolerance: Distance tolerance for matching
            
        Returns:
            node_id
        """
        node_id = self.find_node(position, tolerance)
        if node_id is None:
            node_id = self.add_node(position)
        return node_id
    
    @staticmethod
    def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
        """
        Check if point is inside polygon (ray casting algorithm).
        
        Parameters:
            point: [x, y] point to test
            polygon: Nx2 array of polygon vertices
            
        Returns:
            True if point inside polygon
        """
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
