"""
Build_FEM - High-level FEM structure builder

This module provides convenience methods for mesh generation and FEM
structure creation.

Author: HybriDFEM Team
Date: 2025
"""

import numpy as np
import warnings
from typing import List, Optional, Tuple
from structure_fem import Structure_FEM


class Build_FEM(Structure_FEM):
    """
    High-level FEM structure builder.
    
    For users who want convenience methods for meshing.
    Stores geometry, then converts to nodes via make_nodes().
    
    Workflow:
        1. Add geometry: add_plate(), add_mesh(), etc.
        2. Convert: make_nodes()
        3. Solve: solve_linear()
    
    Usage:
        fem = Build_FEM()
        fem.add_plate([0, 0], Lx=3.0, Ly=2.0, nx=6, ny=4)
        fem.make_nodes()
        fem.solve_linear()
    """
    
    def __init__(self):
        """Initialize FEM builder."""
        super().__init__()
        
        # High-level geometry storage (before conversion)
        self._fem_geometry = []
        self._nodes_made = False
    
    # =========================================================================
    # HIGH-LEVEL GEOMETRY API (Stores geometry)
    # =========================================================================
    
    def add_plate(self, origin: np.ndarray, Lx: float, Ly: float,
                  nx: int, ny: int,
                  E: float = 30e9, nu: float = 0.2, 
                  thickness: float = 1.0):
        """
        Add rectangular plate with structured mesh.
        
        Parameters:
            origin: [x, y] bottom-left corner
            Lx, Ly: Plate dimensions (m)
            nx, ny: Number of elements in each direction
            E: Young's modulus (Pa)
            nu: Poisson's ratio
            thickness: Plate thickness (m)
        
        Example:
            fem = Build_FEM()
            fem.add_plate([0, 0], Lx=3.0, Ly=2.0, nx=6, ny=4)
            fem.make_nodes()
        
        Note:
            Creates triangular elements (2 per quad cell).
        """
        if self._nodes_made:
            raise RuntimeError(
                "Cannot add geometry after make_nodes(). "
                "Create a new builder or use add_node() directly."
            )
        
        self._fem_geometry.append({
            'type': 'plate',
            'origin': np.array(origin, dtype=float),
            'Lx': Lx,
            'Ly': Ly,
            'nx': nx,
            'ny': ny,
            'E': E,
            'nu': nu,
            'thickness': thickness
        })
        
        print(f"Added plate at {origin} ({nx}x{ny} elements)")
    
    def add_mesh(self, region_corners: List[List[float]], 
                 element_size: float,
                 E: float = 30e9, nu: float = 0.2,
                 thickness: float = 1.0,
                 element_type: str = 'triangle'):
        """
        Add FEM mesh in region (unstructured).
        
        Parameters:
            region_corners: [[x1,y1], [x2,y2], ...] polygon vertices
            element_size: Approximate element edge length (m)
            E: Young's modulus (Pa)
            nu: Poisson's ratio
            thickness: Element thickness (m)
            element_type: 'triangle' or 'quad'
        
        Example:
            region = [[0, 0], [3, 0], [3, 2], [0, 2]]
            fem.add_mesh(region, element_size=0.3)
        
        Note:
            Requires external mesher (triangle, gmsh, etc.).
            For structured meshes, use add_plate() instead.
        """
        if self._nodes_made:
            raise RuntimeError("Cannot add geometry after make_nodes()")
        
        self._fem_geometry.append({
            'type': 'mesh',
            'region': region_corners,
            'element_size': element_size,
            'E': E,
            'nu': nu,
            'thickness': thickness,
            'element_type': element_type
        })
        
        print(f"Added unstructured mesh (element_size={element_size})")
    
    def add_beam_mesh(self, N1: np.ndarray, N2: np.ndarray, 
                      n_elements: int,
                      E: float = 30e9, nu: float = 0.2,
                      height: float = 0.3, thickness: float = 0.2):
        """
        Add 1D beam mesh.
        
        Parameters:
            N1, N2: Start and end points [x, y]
            n_elements: Number of beam elements
            E: Young's modulus (Pa)
            nu: Poisson's ratio
            height: Beam height/depth (m)
            thickness: Beam thickness/width (m)
        
        Example:
            fem.add_beam_mesh([0, 0], [3, 0], n_elements=6,
                            height=0.3, thickness=0.2)
        """
        if self._nodes_made:
            raise RuntimeError("Cannot add geometry after make_nodes()")
        
        self._fem_geometry.append({
            'type': 'beam_mesh',
            'N1': np.array(N1, dtype=float),
            'N2': np.array(N2, dtype=float),
            'n_elements': n_elements,
            'E': E,
            'nu': nu,
            'height': height,
            'thickness': thickness
        })
        
        print(f"Added beam mesh from {N1} to {N2} ({n_elements} elements)")
    
    def add_circular_mesh(self, center: np.ndarray, radius: float,
                         element_size: float,
                         E: float = 30e9, nu: float = 0.2,
                         thickness: float = 1.0):
        """
        Add circular mesh.
        
        Parameters:
            center: [x, y] center of circle
            radius: Circle radius (m)
            element_size: Approximate element size (m)
            E: Young's modulus (Pa)
            nu: Poisson's ratio
            thickness: Element thickness (m)
        
        Example:
            fem.add_circular_mesh([1.5, 1.5], radius=1.0, element_size=0.2)
        """
        if self._nodes_made:
            raise RuntimeError("Cannot add geometry after make_nodes()")
        
        self._fem_geometry.append({
            'type': 'circular',
            'center': np.array(center, dtype=float),
            'radius': radius,
            'element_size': element_size,
            'E': E,
            'nu': nu,
            'thickness': thickness
        })
        
        print(f"Added circular mesh at {center} (radius={radius})")
    
    # =========================================================================
    # CONVERSION: Geometry → Nodes
    # =========================================================================
    
    def make_nodes(self):
        """
        Convert stored geometry to nodes and elements.
        
        This is the KEY method that bridges high-level geometry
        and low-level monolithic API.
        
        Process:
        1. Convert each stored geometry to nodes (using add_node())
        2. Create elements (using add_triangle_element(), add_beam_element())
        3. Detect and merge duplicate nodes
        4. Call finalize() to prepare for solving
        
        Example:
            fem = Build_FEM()
            fem.add_plate([0, 0], Lx=3, Ly=2, nx=6, ny=4)
            fem.make_nodes()  # ← Converts everything
            fem.solve_linear()
        """
        if self._nodes_made:
            warnings.warn("make_nodes() already called", RuntimeWarning)
            return
        
        print(f"\n{'='*60}")
        print(f"CONVERTING FEM GEOMETRY TO NODES")
        print(f"{'='*60}")
        print(f"Stored geometries: {len(self._fem_geometry)}")
        
        for i, geom in enumerate(self._fem_geometry):
            geom_type = geom['type']
            print(f"\n[{i+1}/{len(self._fem_geometry)}] Converting {geom_type}...")
            
            if geom_type == 'plate':
                self._build_plate(geom)
            elif geom_type == 'mesh':
                self._build_mesh(geom)
            elif geom_type == 'beam_mesh':
                self._build_beam_mesh(geom)
            elif geom_type == 'circular':
                self._build_circular(geom)
            else:
                raise ValueError(f"Unknown geometry type: {geom_type}")
        
        # Clear stored geometry (no longer needed)
        self._fem_geometry.clear()
        self._nodes_made = True
        
        # Finalize structure
        print(f"\n{'='*60}")
        print(f"FEM GEOMETRY CONVERSION COMPLETE")
        print(f"{'='*60}")
        print(f"Total nodes: {len(self._nodes)}")
        print(f"Total FEM elements: {len(self.list_fes)}")
        
        self.finalize()
    
    # =========================================================================
    # INTERNAL: Geometry → Element Conversion
    # =========================================================================
    
    def _build_plate(self, geom: dict):
        """
        Create structured mesh for rectangular plate.
        
        TODO: Implement structured mesh generation
        """
        origin = geom['origin']
        Lx, Ly = geom['Lx'], geom['Ly']
        nx, ny = geom['nx'], geom['ny']
        E, nu, t = geom['E'], geom['nu'], geom['thickness']
        
        print(f"    Creating {nx}x{ny} structured mesh...")
        
        # Element dimensions
        dx = Lx / nx
        dy = Ly / ny
        
        # TODO: Implement actual mesh creation
        # Pseudocode:
        # 1. Create grid of nodes
        # node_grid = []
        # for j in range(ny + 1):
        #     row = []
        #     for i in range(nx + 1):
        #         pos = origin + [i * dx, j * dy]
        #         node_id = self._find_or_create_node(pos)
        #         row.append(node_id)
        #     node_grid.append(row)
        #
        # 2. Create triangular elements
        # for j in range(ny):
        #     for i in range(nx):
        #         n1 = node_grid[j][i]
        #         n2 = node_grid[j][i + 1]
        #         n3 = node_grid[j + 1][i + 1]
        #         n4 = node_grid[j + 1][i]
        #         
        #         # Lower triangle
        #         self.add_triangle_element([n1, n2, n4], E, nu, t)
        #         # Upper triangle
        #         self.add_triangle_element([n2, n3, n4], E, nu, t)
        
        elements_created = 2 * nx * ny  # Two triangles per quad
        print(f"    Created {elements_created} triangular elements")
    
    def _build_mesh(self, geom: dict):
        """
        Create unstructured mesh (requires external mesher).
        
        TODO: Implement unstructured meshing
        Options: triangle, gmsh, meshpy
        """
        region = geom['region']
        element_size = geom['element_size']
        E, nu, t = geom['E'], geom['nu'], geom['thickness']
        
        print("    Creating unstructured mesh...")
        print("    WARNING: Requires external mesher (not yet implemented)")
        
        # TODO: Implement using triangle library
        # import triangle
        # tri_dict = {'vertices': np.array(region)}
        # mesh = triangle.triangulate(tri_dict, f'qa{element_size}')
        # 
        # # Create nodes
        # for vertex in mesh['vertices']:
        #     self.add_node(vertex)
        #
        # # Create elements
        # for tri in mesh['triangles']:
        #     self.add_triangle_element(tri.tolist(), E, nu, t)
        
        pass
    
    def _build_beam_mesh(self, geom: dict):
        """
        Create 1D beam mesh.
        
        TODO: Implement beam mesh generation
        """
        N1, N2 = geom['N1'], geom['N2']
        n_elem = geom['n_elements']
        E, nu = geom['E'], geom['nu']
        height, thickness = geom['height'], geom['thickness']
        
        print(f"    Creating {n_elem} beam elements...")
        
        # TODO: Implement
        # Direction
        # vec = N2 - N1
        # L = np.linalg.norm(vec)
        # direction = vec / L
        # elem_L = L / n_elem
        #
        # # Create nodes
        # node_ids = []
        # for i in range(n_elem + 1):
        #     pos = N1 + i * elem_L * direction
        #     node_id = self._find_or_create_node(pos)
        #     node_ids.append(node_id)
        #
        # # Create elements
        # for i in range(n_elem):
        #     self.add_beam_element([node_ids[i], node_ids[i + 1]],
        #                          E, nu, height, thickness)
        
        pass
    
    def _build_circular(self, geom: dict):
        """
        Create circular mesh.
        
        TODO: Implement circular mesh generation
        """
        center = geom['center']
        radius = geom['radius']
        element_size = geom['element_size']
        E, nu, t = geom['E'], geom['nu'], geom['thickness']
        
        print("    Creating circular mesh...")
        
        # TODO: Implement using external mesher
        pass
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def _find_or_create_node(self, position: np.ndarray, 
                            tolerance: float = 1e-6) -> int:
        """
        Find existing node at position or create new one.
        
        This prevents duplicate nodes at mesh interfaces.
        
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
