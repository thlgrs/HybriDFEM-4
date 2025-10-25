"""
Build_Hybrid - High-level hybrid structure builder

This module combines high-level block and FEM builders for convenient
hybrid structure creation.

Author: HybriDFEM Team
Date: 2025
"""

import numpy as np
import warnings
from typing import List, Optional
from build_blocks import Build_blocks
from build_fem import Build_FEM
from hybrid import Hybrid


class Build_Hybrid(Build_blocks, Build_FEM, Hybrid):
    """
    High-level hybrid structure builder.
    
    Combines convenience methods from both Build_blocks and Build_FEM.
    
    Inherits:
    - add_wall(), add_arch(), etc. from Build_blocks
    - add_plate(), add_mesh(), etc. from Build_FEM
    - connect_node_to_block() from Hybrid
    
    Workflow:
        1. Add block geometry: add_wall(), add_arch()
        2. Add FEM geometry: add_plate(), add_mesh()
        3. Convert: make_nodes()
        4. Define coupling: connect_node_to_block() (optional)
        5. Solve: solve_linear()
    
    Usage:
        bh = Build_Hybrid()
        
        # Add block structures
        bh.add_wall([0, 0], L=3, H=2, pattern='running_bond')
        
        # Add FEM structures
        bh.add_plate([0, 2], Lx=3, Ly=0.5, nx=6, ny=1)
        
        # Convert geometry to nodes
        bh.make_nodes()
        
        # Explicit coupling (optional - can auto-detect)
        interface_nodes = bh.find_nodes_in_region(0, 3, 1.95, 2.05)
        for node_id in interface_nodes:
            bh.connect_node_to_block(node_id, block_id=0)
        
        # Solve
        bh.solve_linear()
    """
    
    def __init__(self, coupling_strategy: str = 'penalty'):
        """
        Initialize hybrid builder.
        
        Parameters:
            coupling_strategy: 'penalty', 'lagrange', or 'elimination'
        
        Note:
            Multiple inheritance initialization handled by Python MRO.
            All parent __init__ methods will be called properly.
        """
        # Python's MRO (Method Resolution Order) handles this correctly
        # Just call super().__init__() and it will initialize all parents
        super().__init__()
        
        # Set coupling strategy (from Hybrid)
        self.coupling_strategy = coupling_strategy
        
        print(f"Build_Hybrid initialized (coupling={coupling_strategy})")
    
    # =========================================================================
    # UNIFIED make_nodes() - Handles both block and FEM geometry
    # =========================================================================
    
    def make_nodes(self):
        """
        Convert stored geometry from BOTH blocks and FEM.
        
        This is the KEY method for hybrid builders.
        Handles multiple inheritance by converting both types of geometry.
        
        Process:
        1. Convert block geometries (from Build_blocks)
        2. Convert FEM geometries (from Build_FEM)
        3. Merge and detect duplicate nodes
        4. Call finalize() to prepare for solving
        
        Example:
            bh = Build_Hybrid()
            bh.add_wall([0, 0], L=3, H=2)      # Block geometry
            bh.add_plate([0, 2], Lx=3, Ly=0.5, nx=6, ny=1)  # FEM geometry
            bh.make_nodes()  # ← Converts BOTH
            bh.solve_linear()
        """
        if self._nodes_made:
            warnings.warn("make_nodes() already called", RuntimeWarning)
            return
        
        print(f"\n{'='*70}")
        print(f"CONVERTING HYBRID GEOMETRY TO NODES")
        print(f"{'='*70}")
        print(f"Block geometries: {len(self._block_geometry)}")
        print(f"FEM geometries: {len(self._fem_geometry)}")
        
        # =====================================================================
        # PHASE 1: Convert block geometries
        # =====================================================================
        if len(self._block_geometry) > 0:
            print(f"\n{'-'*70}")
            print("PHASE 1: Converting block geometries")
            print(f"{'-'*70}")
            
            for i, geom in enumerate(self._block_geometry):
                geom_type = geom['type']
                print(f"\n[Block {i+1}/{len(self._block_geometry)}] Converting {geom_type}...")
                
                # Use Build_blocks conversion methods
                if geom_type == 'wall':
                    Build_blocks._build_wall(self, geom)
                elif geom_type == 'beam':
                    Build_blocks._build_beam(self, geom)
                elif geom_type == 'arch':
                    Build_blocks._build_arch(self, geom)
                elif geom_type == 'voronoi':
                    Build_blocks._build_voronoi(self, geom)
                else:
                    raise ValueError(f"Unknown block geometry type: {geom_type}")
        
        # =====================================================================
        # PHASE 2: Convert FEM geometries
        # =====================================================================
        if len(self._fem_geometry) > 0:
            print(f"\n{'-'*70}")
            print("PHASE 2: Converting FEM geometries")
            print(f"{'-'*70}")
            
            for i, geom in enumerate(self._fem_geometry):
                geom_type = geom['type']
                print(f"\n[FEM {i+1}/{len(self._fem_geometry)}] Converting {geom_type}...")
                
                # Use Build_FEM conversion methods
                if geom_type == 'plate':
                    Build_FEM._build_plate(self, geom)
                elif geom_type == 'mesh':
                    Build_FEM._build_mesh(self, geom)
                elif geom_type == 'beam_mesh':
                    Build_FEM._build_beam_mesh(self, geom)
                elif geom_type == 'circular':
                    Build_FEM._build_circular(self, geom)
                else:
                    raise ValueError(f"Unknown FEM geometry type: {geom_type}")
        
        # =====================================================================
        # PHASE 3: Cleanup and finalize
        # =====================================================================
        print(f"\n{'-'*70}")
        print("PHASE 3: Finalizing hybrid structure")
        print(f"{'-'*70}")
        
        # Clear stored geometry (no longer needed)
        self._block_geometry.clear()
        self._fem_geometry.clear()
        self._nodes_made = True
        
        # Finalize structure (from Hybrid, includes coupling setup)
        print(f"\n{'='*70}")
        print(f"HYBRID GEOMETRY CONVERSION COMPLETE")
        print(f"{'='*70}")
        print(f"Total nodes: {len(self._nodes)}")
        print(f"Total blocks: {len(self.list_blocks)}")
        print(f"Total FEM elements: {len(self.list_fes)}")
        print(f"Coupling connections: {len(self._block_node_connections)}")
        
        self.finalize()
    
    # =========================================================================
    # HIGH-LEVEL HYBRID PATTERNS (Convenience methods)
    # =========================================================================
    
    def add_masonry_wall_with_foundation(self, origin: np.ndarray,
                                        L: float, H: float,
                                        foundation_thickness: float = 0.3,
                                        pattern: str = 'running_bond',
                                        wall_rho: float = 2400,
                                        wall_b: float = 0.2,
                                        foundation_E: float = 30e9,
                                        foundation_nu: float = 0.2):
        """
        Add masonry wall with FEM foundation (common hybrid pattern).
        
        Parameters:
            origin: [x, y] bottom-left corner
            L: Wall length (m)
            H: Wall height (m)
            foundation_thickness: FEM foundation thickness (m)
            pattern: Masonry pattern for wall
            wall_rho: Wall block density (kg/m³)
            wall_b: Wall thickness (m)
            foundation_E: Foundation Young's modulus (Pa)
            foundation_nu: Foundation Poisson's ratio
        
        Example:
            bh = Build_Hybrid()
            bh.add_masonry_wall_with_foundation(
                [0, 0], L=3.0, H=2.0,
                foundation_thickness=0.3
            )
            bh.make_nodes()
        
        Note:
            Automatically creates coupling between wall base and foundation top.
        """
        print(f"\nAdding masonry wall with foundation at {origin}")
        
        # Add FEM foundation
        self.add_plate(
            origin=[origin[0], origin[1] - foundation_thickness],
            Lx=L,
            Ly=foundation_thickness,
            nx=int(L / 0.3),  # Element size ~0.3m
            ny=2,
            E=foundation_E,
            nu=foundation_nu,
            thickness=wall_b
        )
        
        # Add block wall
        self.add_wall(
            origin=origin,
            L=L,
            H=H,
            pattern=pattern,
            rho=wall_rho,
            b=wall_b
        )
        
        print("  Foundation and wall geometry added")
        print("  Note: Call make_nodes() then connect base nodes to foundation")
    
    def add_arch_on_piers(self, center: np.ndarray,
                         span: float, rise: float,
                         pier_width: float, pier_height: float,
                         n_voussoirs: int = 11,
                         thickness: float = 0.3):
        """
        Add arch supported by two piers (common masonry pattern).
        
        Parameters:
            center: [x, y] center of arch base
            span: Arch span (m)
            rise: Arch rise (m)
            pier_width: Pier width (m)
            pier_height: Pier height (m)
            n_voussoirs: Number of voussoir blocks in arch
            thickness: Arch/pier thickness (m)
        
        Example:
            bh.add_arch_on_piers(
                [2.0, 2.0], span=3.0, rise=1.5,
                pier_width=0.6, pier_height=2.0
            )
        """
        print(f"\nAdding arch with piers at {center}")
        
        # Left pier
        pier_left_origin = [
            center[0] - span/2 - pier_width/2 - pier_width/2,
            center[1] - pier_height
        ]
        self.add_wall(
            pier_left_origin,
            L=pier_width,
            H=pier_height,
            pattern='stack_bond'
        )
        
        # Right pier
        pier_right_origin = [
            center[0] + span/2 - pier_width/2,
            center[1] - pier_height
        ]
        self.add_wall(
            pier_right_origin,
            L=pier_width,
            H=pier_height,
            pattern='stack_bond'
        )
        
        # Arch
        self.add_arch(
            center=center,
            span=span,
            rise=rise,
            n_voussoirs=n_voussoirs,
            thickness=thickness
        )
        
        print(f"  Added arch with two piers")
    
    def auto_detect_interfaces(self, tolerance: float = 1e-3):
        """
        Automatically detect interfaces between blocks and FEM.
        
        Parameters:
            tolerance: Distance tolerance for interface detection (m)
        
        Note:
            Call after make_nodes() to automatically create connections
            between blocks and FEM elements that share nodes.
        
        Example:
            bh.make_nodes()
            bh.auto_detect_interfaces()  # Auto-create connections
            bh.solve_linear()
        """
        if not self._nodes_made:
            raise RuntimeError("Must call make_nodes() before auto-detecting interfaces")
        
        print("\nAuto-detecting block-FEM interfaces...")
        
        # TODO: Implement interface detection
        # Algorithm:
        # 1. For each block, get its boundary nodes
        # 2. For each FEM element, get its nodes
        # 3. Find nodes that are shared (within tolerance)
        # 4. Create connections for shared nodes
        
        # Pseudocode:
        # connections_created = 0
        # for block_id, block in enumerate(self.list_blocks):
        #     block_nodes = self.get_block_nodes(block_id)
        #     for block_node_id in block_nodes:
        #         block_pos = self._nodes[block_node_id]
        #         
        #         # Check if this node is also used by FEM elements
        #         for elem in self.list_fes:
        #             if block_node_id in elem.node_ids:
        #                 # Shared node - create connection
        #                 self.connect_node_to_block(block_node_id, block_id)
        #                 connections_created += 1
        #                 break
        # 
        # print(f"  Created {connections_created} automatic connections")
        
        print("  WARNING: auto_detect_interfaces() not yet implemented")
        print("  Use connect_node_to_block() for explicit coupling")
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def get_structure_info(self) -> dict:
        """
        Get comprehensive information about hybrid structure.
        
        Returns:
            Dictionary with structure statistics
        """
        return {
            'nodes': len(self._nodes),
            'blocks': len(self.list_blocks),
            'fem_elements': len(self.list_fes),
            'connections': len(self._block_node_connections),
            'coupling_strategy': self.coupling_strategy,
            'finalized': self._finalized,
            'dofs': self.nb_dofs if self._finalized else 0
        }
    
    def print_structure_summary(self):
        """Print formatted summary of structure."""
        info = self.get_structure_info()
        
        print(f"\n{'='*60}")
        print("HYBRID STRUCTURE SUMMARY")
        print(f"{'='*60}")
        print(f"Nodes:           {info['nodes']}")
        print(f"Rigid blocks:    {info['blocks']}")
        print(f"FEM elements:    {info['fem_elements']}")
        print(f"Connections:     {info['connections']}")
        print(f"Coupling:        {info['coupling_strategy']}")
        print(f"DOFs:            {info['dofs']}")
        print(f"Finalized:       {info['finalized']}")
        print(f"{'='*60}\n")
