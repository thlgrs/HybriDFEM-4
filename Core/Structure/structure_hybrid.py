"""
Hybrid - Low-level hybrid FEM-DEM structures

This module combines FEM and rigid block structures with explicit coupling.

Author: HybriDFEM Team
Date: 2025
"""

import numpy as np
import warnings
from typing import List, Optional, Dict
from structure_block import Structure_block
from structure_fem import Structure_FEM


class Hybrid(Structure_block, Structure_FEM):
    """
    Low-level hybrid FEM-DEM structures.
    
    Combines FEM and rigid block capabilities with explicit coupling control.
    
    Inherits:
    - add_node() from Structure_2D
    - add_triangle_element(), add_beam_element() from Structure_FEM
    - add_rigid_block_by_nodes() from Structure_block
    
    Adds:
    - connect_node_to_block() - explicit hybrid coupling
    
    Usage:
        hybrid = Hybrid()
        
        # Create FEM nodes and elements
        n0 = hybrid.add_node([0, 0])
        n1 = hybrid.add_node([1, 0])
        n2 = hybrid.add_node([0, 1])
        hybrid.add_triangle_element([n0, n1, n2], E=30e9, nu=0.2)
        
        # Create rigid block
        n3 = hybrid.add_node([1, 0])
        n4 = hybrid.add_node([2, 0])
        n5 = hybrid.add_node([2, 1])
        n6 = hybrid.add_node([1, 1])
        block_id = hybrid.add_rigid_block_by_nodes(
            [n3, n4, n5, n6], 
            ref_point=[1.5, 0.5]
        )
        
        # Explicit coupling
        hybrid.connect_node_to_block(n1, block_id)
        
        hybrid.finalize()
        hybrid.solve_linear()
    """
    
    def __init__(self, coupling_strategy: str = 'penalty'):
        """
        Initialize hybrid structure.
        
        Parameters:
            coupling_strategy: Coupling method:
                - 'penalty': Penalty method (soft constraint)
                - 'lagrange': Lagrange multipliers (hard constraint)
                - 'elimination': Constraint elimination (DOF reduction)
        """
        super().__init__()
        
        self.coupling_strategy = coupling_strategy
        self._block_node_connections: List[Dict] = []
        
        # Hybrid-specific storage
        self.list_hybrid_cfs = []  # Hybrid contact faces
    
    # =========================================================================
    # HYBRID-SPECIFIC: Coupling (ONLY method unique to Hybrid)
    # =========================================================================
    
    def connect_node_to_block(self, node_id: int, block_id: int,
                             coupling_type: Optional[str] = None):
        """
        Connect FEM node to rigid block (explicit coupling).
        
        This is the KEY method that makes Hybrid different from its parents.
        It explicitly defines which FEM nodes follow rigid block motion.
        
        Parameters:
            node_id: FEM node ID to connect
            block_id: Rigid block ID
            coupling_type: Override global coupling_strategy
                - 'rigid': Kinematic (node follows block exactly)
                - 'penalty': Penalty springs
                - 'lagrange': Lagrange multipliers
        
        Example:
            # Connect interface nodes to block
            interface_nodes = hybrid.find_nodes_in_region(0.9, 1.1, 0, 2)
            for node_id in interface_nodes:
                hybrid.connect_node_to_block(node_id, block_id=0)
        
        Note:
            Must be called before finalize().
            Coupling is enforced during system assembly.
        """
        if self._finalized:
            raise RuntimeError(
                "Cannot add connections after finalize(). "
                "Call before finalization."
            )
        
        if node_id not in self._nodes:
            raise ValueError(f"Node {node_id} doesn't exist")
        
        if block_id >= len(self.list_blocks):
            raise ValueError(f"Block {block_id} doesn't exist")
        
        if coupling_type is None:
            coupling_type = self.coupling_strategy
        
        self._block_node_connections.append({
            'node_id': node_id,
            'block_id': block_id,
            'type': coupling_type
        })
        
        print(f"Connected node {node_id} to block {block_id} (type={coupling_type})")
    
    def disconnect_node_from_block(self, node_id: int, block_id: int):
        """
        Remove explicit connection (if added by mistake).
        
        Parameters:
            node_id: Node ID
            block_id: Block ID
        """
        self._block_node_connections = [
            conn for conn in self._block_node_connections
            if not (conn['node_id'] == node_id and conn['block_id'] == block_id)
        ]
    
    def get_connections(self) -> List[Dict]:
        """
        Get all node-block connections.
        
        Returns:
            List of connection dictionaries
        """
        return self._block_node_connections.copy()
    
    # =========================================================================
    # OVERRIDE: Finalize with coupling setup
    # =========================================================================
    
    def finalize(self):
        """
        Finalize hybrid structure with coupling.
        
        Extends parent finalize() to set up hybrid coupling.
        """
        # Call parent finalization
        super().finalize()
        
        print(f"\nSetting up hybrid coupling ({self.coupling_strategy})...")
        print(f"  Connections: {len(self._block_node_connections)}")
        
        # Setup coupling based on strategy
        if self.coupling_strategy == 'elimination':
            self._setup_constraint_elimination()
        elif self.coupling_strategy == 'penalty':
            self._setup_penalty_coupling()
        elif self.coupling_strategy == 'lagrange':
            self._setup_lagrange_coupling()
        else:
            raise ValueError(f"Unknown coupling strategy: {self.coupling_strategy}")
    
    def _setup_constraint_elimination(self):
        """
        Setup constraint elimination coupling.
        
        TODO: Implement constraint elimination
        - Eliminate DOFs of connected nodes
        - Express node motion in terms of block DOFs
        - Build reduced system
        """
        print("  Using constraint elimination (DOF reduction)")
        # TODO: Implement
        # This is where your uploaded GeneralizedCoupledSystem code would integrate
        pass
    
    def _setup_penalty_coupling(self):
        """
        Setup penalty method coupling.
        
        TODO: Implement penalty coupling
        - Add penalty springs between nodes and blocks
        - No DOF elimination (full system)
        """
        print("  Using penalty method")
        # TODO: Implement
        pass
    
    def _setup_lagrange_coupling(self):
        """
        Setup Lagrange multiplier coupling.
        
        TODO: Implement Lagrange multipliers
        - Add constraint equations
        - Augment system with multipliers
        """
        print("  Using Lagrange multipliers")
        # TODO: Implement
        pass
    
    # =========================================================================
    # ASSEMBLY (Combines both parents)
    # =========================================================================
    
    def get_K_str(self):
        """
        Assemble global stiffness matrix from FEM, blocks, and coupling.
        """
        if not self._finalized:
            raise RuntimeError("Must call finalize() before assembly")
        
        print("\nAssembling hybrid stiffness matrix...")
        
        # Initialize global stiffness
        self.K = np.zeros((self.nb_dofs, self.nb_dofs))
        
        # Assemble FEM contribution
        print("  FEM stiffness...")
        self._stiffness_fem()
        
        # Assemble block contribution
        print("  Block stiffness...")
        self._stiffness_block()
        
        # Assemble coupling contribution
        print("  Coupling stiffness...")
        self._stiffness_hybrid()
        
        print(f"  Assembly complete (size={self.nb_dofs}x{self.nb_dofs})")
        
        return self.K
    
    def _stiffness_hybrid(self):
        """
        Assemble hybrid coupling stiffness.
        
        TODO: Implement coupling stiffness based on strategy
        - Penalty: Add spring stiffness
        - Lagrange: Add constraint terms
        - Elimination: Handled during setup (no addition here)
        """
        if self.coupling_strategy == 'penalty':
            # TODO: Add penalty springs
            pass
        elif self.coupling_strategy == 'lagrange':
            # TODO: Add constraint terms
            pass
        # Elimination: no action needed here
    
    def get_P_r(self):
        """
        Assemble residual force vector (hybrid).
        """
        if not self._finalized:
            raise RuntimeError("Must call finalize() before assembly")
        
        # TODO: Implement hybrid residual
        # Combine FEM and block residuals
        # Add coupling forces if needed
        
        self.P_r = self.P.copy()
        
        return self.P_r
    
    # =========================================================================
    # LEGACY COMPATIBILITY
    # =========================================================================
    
    def make_hybrid_couplings(self):
        """
        Create hybrid couplings (LEGACY method).
        
        .. deprecated:: 4.0
            Use connect_node_to_block() for explicit control.
        """
        warnings.warn(
            "make_hybrid_couplings() is deprecated. "
            "Use connect_node_to_block() for explicit coupling.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # TODO: Auto-detect interfaces and create connections
        # This is the old automatic coupling detection
        pass
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def get_coupling_info(self) -> Dict:
        """
        Get information about hybrid coupling.
        
        Returns:
            Dictionary with coupling statistics
        """
        return {
            'strategy': self.coupling_strategy,
            'n_connections': len(self._block_node_connections),
            'n_fem_elements': len(self.list_fes),
            'n_blocks': len(self.list_blocks),
            'n_nodes': len(self._nodes)
        }
