"""
Structure_block - Low-level rigid block structures

This module provides precise control over rigid block creation and assembly.

Author: HybriDFEM Team
Date: 2025
"""

import numpy as np
import warnings
from typing import List, Optional
from structure_2d import Structure_2D


class Structure_block(Structure_2D):
    """
    Low-level rigid block structures.
    
    For users who want precise control over rigid block creation using
    existing nodes.
    
    Usage:
        sb = Structure_block()
        n0 = sb.add_node([0, 0])
        n1 = sb.add_node([1, 0])
        n2 = sb.add_node([1, 1])
        n3 = sb.add_node([0, 1])
        block_id = sb.add_rigid_block_by_nodes(
            [n0, n1, n2, n3], 
            ref_point=[0.5, 0.5],
            rho=2400,
            b=0.2
        )
        sb.finalize()
        sb.solve_linear()
    """
    
    def __init__(self):
        """Initialize block structure."""
        super().__init__()
        
        # Legacy storage (for compatibility)
        self.list_blocks = []
        self.list_cfs = []  # Contact faces
    
    # =========================================================================
    # BLOCK CREATION (Monolithic API)
    # =========================================================================
    
    def add_rigid_block_by_nodes(self, node_ids: List[int], 
                                  ref_point: np.ndarray,
                                  rho: float = 2400, 
                                  b: float = 1.0) -> int:
        """
        Add rigid block using existing nodes (monolithic API).
        
        Parameters:
            node_ids: List of node IDs forming block vertices (CCW order)
            ref_point: [x, y] reference point for rigid body motion
            rho: Density (kg/m³)
            b: Thickness (m)
            
        Returns:
            block_id: Integer ID of created block
            
        Example:
            sb = Structure_block()
            n0 = sb.add_node([0, 0])
            n1 = sb.add_node([1, 0])
            n2 = sb.add_node([1, 1])
            n3 = sb.add_node([0, 1])
            block_id = sb.add_rigid_block_by_nodes(
                [n0, n1, n2, n3], 
                ref_point=[0.5, 0.5]
            )
        
        Note:
            Vertices should be ordered counter-clockwise.
            Reference point is typically the centroid.
        """
        # Validate nodes exist
        for nid in node_ids:
            if nid not in self._nodes:
                raise ValueError(f"Node {nid} doesn't exist. Add it with add_node() first.")
        
        # Get vertex positions
        vertices = np.array([self._nodes[nid] for nid in node_ids])
        
        # TODO: Create Block object (import from existing HybriDFEM code)
        # from your_existing_code import Block
        # block = Block(vertices, ref_point, rho, b)
        # block.node_ids = node_ids  # Store connectivity
        
        # Placeholder: create simple block representation
        block = {
            'node_ids': node_ids,
            'vertices': vertices,
            'ref_point': np.array(ref_point, dtype=float),
            'rho': rho,
            'b': b
        }
        
        block_id = len(self.list_blocks)
        self.list_blocks.append(block)
        self._elements.append(block)  # Unified storage
        
        return block_id
    
    def get_block_nodes(self, block_id: int) -> List[int]:
        """
        Get node IDs belonging to a block.
        
        Parameters:
            block_id: Block ID
            
        Returns:
            List of node IDs
        """
        if block_id >= len(self.list_blocks):
            raise ValueError(f"Block {block_id} doesn't exist")
        
        block = self.list_blocks[block_id]
        
        # TODO: Handle different block representations
        if isinstance(block, dict):
            return block['node_ids']
        else:
            # Handle Block object
            return block.node_ids
    
    # =========================================================================
    # LEGACY API (Deprecated)
    # =========================================================================
    
    def add_block(self, vertices: np.ndarray, rho: float = 2400, 
                  b: float = 1.0, **kwargs) -> int:
        """
        Add rigid block by vertices (LEGACY method).
        
        .. deprecated:: 4.0
            Use add_rigid_block_by_nodes() instead for explicit control.
            
        Parameters:
            vertices: Nx2 array of vertex positions
            rho: Density
            b: Thickness
            
        Returns:
            block_id
        """
        warnings.warn(
            "add_block() is deprecated. Use add_rigid_block_by_nodes() for "
            "explicit node control.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Convert to monolithic API
        node_ids = []
        for v in vertices:
            # Check if node already exists
            nid = self.find_node(v, tolerance=1e-6)
            if nid is None:
                nid = self.add_node(v)
            node_ids.append(nid)
        
        # Compute reference point (centroid)
        ref_point = np.mean(vertices, axis=0)
        
        return self.add_rigid_block_by_nodes(node_ids, ref_point, rho, b)
    
    # =========================================================================
    # ASSEMBLY
    # =========================================================================
    
    def _stiffness_block(self):
        """
        Assemble block stiffness matrix.
        
        TODO: Implement block stiffness assembly
        - Compute block mass properties
        - Build element stiffness matrices
        - Assemble into global K
        """
        # TODO: Implement actual stiffness assembly
        # For now, placeholder
        print("  Assembling block stiffness...")
        pass
    
    def get_K_str(self):
        """Assemble global stiffness matrix."""
        if not self._finalized:
            raise RuntimeError("Must call finalize() before assembly")
        
        # Initialize global stiffness
        self.K = np.zeros((self.nb_dofs, self.nb_dofs))
        
        # Assemble block contributions
        self._stiffness_block()
        
        return self.K
    
    def get_P_r(self):
        """
        Assemble residual force vector.
        
        TODO: Implement residual assembly
        - Compute internal forces
        - Subtract from external forces
        """
        if not self._finalized:
            raise RuntimeError("Must call finalize() before assembly")
        
        # TODO: Implement actual residual assembly
        # For now, residual = external forces
        self.P_r = self.P.copy()
        
        return self.P_r
    
    # =========================================================================
    # CONTACT/INTERFACE METHODS (Legacy)
    # =========================================================================
    
    def make_cfs(self, auto_detect: bool = True, nb_cps: int = 5):
        """
        Create contact faces (LEGACY method).
        
        TODO: Implement contact face detection
        - Detect block-block interfaces
        - Create contact point pairs
        """
        # TODO: Implement contact face creation
        print(f"Creating contact faces (auto={auto_detect}, nb_cps={nb_cps})...")
        pass
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def get_block_count(self) -> int:
        """Get total number of blocks."""
        return len(self.list_blocks)
