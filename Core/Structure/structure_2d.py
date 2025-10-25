"""
Structure_2D - Base class for 2D structural analysis

This module provides the low-level monolithic API for precise control
over nodes, DOFs, and system assembly.

Author: HybriDFEM Team
Date: 2025
"""

from abc import ABC, abstractmethod
import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple


class Structure_2D(ABC):
    """
    Abstract base class for 2D structural analysis.
    
    Provides low-level monolithic API:
    - Node management (concrete methods)
    - DOF system management (concrete methods)
    - Finalization workflow (concrete methods)
    
    Requires children to implement:
    - get_K_str() - Stiffness assembly
    - get_P_r() - Residual assembly
    
    Usage (via child class):
        st = Structure_FEM()  # or Structure_block or Hybrid
        n0 = st.add_node([0, 0])
        n1 = st.add_node([1, 0])
        st.add_triangle_element([n0, n1, n2], E=30e9, nu=0.2)
        st.finalize()
        st.solve_linear()
    """
    
    def __init__(self):
        """Initialize structure with empty state."""
        # =====================================================================
        # MONOLITHIC STORAGE (Primary - NEW API)
        # =====================================================================
        self._nodes: Dict[int, np.ndarray] = {}  # {node_id: position}
        self._elements: List = []                 # All elements (unified)
        self._node_counter: int = 0
        self._finalized: bool = False
        
        # =====================================================================
        # LEGACY STORAGE (Compatibility with old code)
        # =====================================================================
        self.list_nodes: List[np.ndarray] = []  # Built by finalize()
        
        # =====================================================================
        # DOF SYSTEM (Built by finalize())
        # =====================================================================
        self.nb_dofs: int = 0
        self.U: Optional[np.ndarray] = None      # Displacement vector
        self.P: Optional[np.ndarray] = None      # External force vector
        self.P_fixed: Optional[np.ndarray] = None  # Fixed forces
        self.dof_fix: np.ndarray = np.array([], dtype=int)  # Fixed DOFs
        self.dof_free: Optional[np.ndarray] = None  # Free DOFs
        self.nb_dof_fix: int = 0
        self.nb_dof_free: int = 0
        
        # =====================================================================
        # SYSTEM MATRICES (Built during solve)
        # =====================================================================
        self.K: Optional[np.ndarray] = None      # Stiffness matrix
        self.P_r: Optional[np.ndarray] = None    # Residual force vector
        self.M: Optional[np.ndarray] = None      # Mass matrix
    
    # =========================================================================
    # NODE MANAGEMENT (Concrete - all children use)
    # =========================================================================
    
    def add_node(self, position: np.ndarray) -> int:
        """
        Add node at position (monolithic API).
        
        Parameters:
            position: [x, y] coordinates
            
        Returns:
            node_id: Integer ID of created node (0-indexed)
            
        Example:
            n0 = structure.add_node([0.0, 0.0])
            n1 = structure.add_node([1.0, 0.0])
            print(f"Created nodes {n0} and {n1}")
        
        Note:
            If called after finalize(), will set _finalized=False
            requiring finalize() to be called again before solving.
        """
        if self._finalized:
            warnings.warn(
                "Adding node after finalize() requires calling finalize() again",
                RuntimeWarning,
                stacklevel=2
            )
            self._finalized = False
        
        node_id = self._node_counter
        self._nodes[node_id] = np.array(position, dtype=float)
        self._node_counter += 1
        return node_id
    
    def get_all_nodes(self) -> Dict[int, np.ndarray]:
        """
        Get all nodes as dictionary.
        
        Returns:
            Dictionary mapping node_id to position [x, y]
            
        Example:
            nodes = structure.get_all_nodes()
            for nid, pos in nodes.items():
                print(f"Node {nid}: {pos}")
        """
        return {nid: pos.copy() for nid, pos in self._nodes.items()}
    
    def find_node(self, position: np.ndarray, tolerance: float = 1e-6) -> Optional[int]:
        """
        Find node at given position.
        
        Parameters:
            position: [x, y] coordinates to search
            tolerance: Distance tolerance for matching
            
        Returns:
            node_id if found, None otherwise
            
        Example:
            node_id = structure.find_node([1.0, 0.5])
            if node_id is not None:
                structure.apply_force_to_node(node_id, dof=1, value=-1000)
        """
        position = np.array(position, dtype=float)
        for nid, pos in self._nodes.items():
            if np.linalg.norm(pos - position) < tolerance:
                return nid
        return None
    
    def find_nodes_in_region(self, xmin: float, xmax: float,
                            ymin: float, ymax: float) -> List[int]:
        """
        Find all nodes within rectangular region.
        
        Parameters:
            xmin, xmax: X-coordinate bounds
            ymin, ymax: Y-coordinate bounds
            
        Returns:
            List of node IDs within region
            
        Example:
            top_nodes = structure.find_nodes_in_region(0, 3, 2.9, 3.1)
            for node_id in top_nodes:
                structure.fix_node_by_id(node_id, [0, 1, 2])
        """
        nodes_in_region = []
        for nid, pos in self._nodes.items():
            if xmin <= pos[0] <= xmax and ymin <= pos[1] <= ymax:
                nodes_in_region.append(nid)
        return nodes_in_region
    
    def modify_node_position(self, node_id: int, new_position: np.ndarray):
        """
        Modify position of existing node.
        
        Parameters:
            node_id: ID of node to modify
            new_position: New [x, y] coordinates
            
        Warning:
            This invalidates element stiffness matrices.
            Must call finalize() again before solving.
        """
        if node_id not in self._nodes:
            raise ValueError(f"Node {node_id} doesn't exist")
        
        self._nodes[node_id] = np.array(new_position, dtype=float)
        self._finalized = False
        warnings.warn(
            "Node modified - call finalize() before solving",
            RuntimeWarning,
            stacklevel=2
        )
    
    def get_node_count(self) -> int:
        """Get total number of nodes."""
        return len(self._nodes)
    
    # =========================================================================
    # FINALIZATION (Concrete - all children use)
    # =========================================================================
    
    def finalize(self):
        """
        Prepare structure for solving.
        
        This method:
        1. Builds legacy list_nodes for compatibility
        2. Constructs DOF system (U, P, dof_fix, dof_free)
        3. Initializes system matrices
        
        Must be called after all nodes/elements added and before solving.
        
        Example:
            structure.add_node([0, 0])
            structure.add_triangle_element([n0, n1, n2], ...)
            structure.finalize()  # ← Must call this
            structure.solve_linear()
        """
        if self._finalized:
            warnings.warn("Structure already finalized", RuntimeWarning)
            return
        
        print(f"Finalizing structure with {len(self._nodes)} nodes...")
        
        # Build legacy list_nodes (for compatibility with old code)
        n_nodes = len(self._nodes)
        self.list_nodes = [self._nodes[i] for i in range(n_nodes)]
        
        # Build DOF system (3 DOFs per node: u, v, theta)
        self.nb_dofs = 3 * n_nodes
        self.U = np.zeros(self.nb_dofs, dtype=float)
        self.P = np.zeros(self.nb_dofs, dtype=float)
        self.P_fixed = np.zeros(self.nb_dofs, dtype=float)
        
        # Initially all DOFs are free
        self.dof_free = np.arange(self.nb_dofs, dtype=int)
        self.nb_dof_free = self.nb_dofs
        self.nb_dof_fix = 0
        
        # Initialize matrices (will be built during solve)
        self.K = None
        self.P_r = None
        self.M = None
        
        self._finalized = True
        print("Finalization complete.")
    
    def dofs_defined(self):
        """
        Check if DOFs are defined (legacy compatibility).
        
        Raises:
            RuntimeError if structure not finalized
        """
        if not self._finalized:
            raise RuntimeError(
                "Structure not finalized. Call finalize() before solving."
            )
    
    # =========================================================================
    # FORCE/BC APPLICATION (Concrete)
    # =========================================================================
    
    def apply_force_to_node(self, node_id: int, dof: int, value: float):
        """
        Apply force to specific node by ID.
        
        Parameters:
            node_id: Node ID (0-indexed)
            dof: DOF index (0=u, 1=v, 2=theta)
            value: Force magnitude
            
        Example:
            structure.apply_force_to_node(5, dof=1, value=-1000)  # Vertical
        """
        if not self._finalized:
            raise RuntimeError("Must call finalize() before applying forces")
        
        if node_id not in self._nodes:
            raise ValueError(f"Node {node_id} doesn't exist")
        
        global_dof = 3 * node_id + dof
        self.P[global_dof] = value
    
    def fix_node_by_id(self, node_id: int, dofs: List[int]):
        """
        Fix specific DOFs of a node.
        
        Parameters:
            node_id: Node ID
            dofs: List of DOF indices to fix (0=u, 1=v, 2=theta)
            
        Example:
            # Fix all DOFs of node 0
            structure.fix_node_by_id(0, [0, 1, 2])
            
            # Fix only vertical displacement of node 5
            structure.fix_node_by_id(5, [1])
        """
        if not self._finalized:
            raise RuntimeError("Must call finalize() before fixing DOFs")
        
        if node_id not in self._nodes:
            raise ValueError(f"Node {node_id} doesn't exist")
        
        for dof in dofs:
            if dof not in [0, 1, 2]:
                raise ValueError(f"DOF must be 0, 1, or 2, got {dof}")
            
            global_dof = 3 * node_id + dof
            if global_dof not in self.dof_fix:
                self.dof_fix = np.append(self.dof_fix, global_dof)
        
        # Update free DOFs
        self.dof_free = np.setdiff1d(np.arange(self.nb_dofs), self.dof_fix)
        self.nb_dof_fix = len(self.dof_fix)
        self.nb_dof_free = len(self.dof_free)
    
    # =========================================================================
    # LEGACY COMPATIBILITY (Wrapper methods - deprecated)
    # =========================================================================
    
    def loadNode(self, node_location, dof: int, value: float, fixed: bool = False):
        """
        Apply force to node (LEGACY method).
        
        .. deprecated:: 4.0
            Use apply_force_to_node() instead.
            
        Parameters:
            node_location: Node ID (int) or position (array-like)
            dof: DOF index
            value: Force value
            fixed: If True, also set as fixed force
        """
        warnings.warn(
            "loadNode() is deprecated. Use apply_force_to_node().",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Determine if node_location is ID or position
        if isinstance(node_location, (int, np.integer)):
            node_id = node_location
        else:
            node_id = self.find_node(node_location)
            if node_id is None:
                raise ValueError(f"No node found at {node_location}")
        
        self.apply_force_to_node(node_id, dof, value)
        
        if fixed:
            global_dof = 3 * node_id + dof
            self.P_fixed[global_dof] = value
    
    def fixNode(self, node_location, dofs: List[int]):
        """
        Fix node DOFs (LEGACY method).
        
        .. deprecated:: 4.0
            Use fix_node_by_id() instead.
            
        Parameters:
            node_location: Node ID (int) or position (array-like), or list of locations
            dofs: List of DOF indices to fix
        """
        warnings.warn(
            "fixNode() is deprecated. Use fix_node_by_id().",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Handle multiple nodes
        if isinstance(node_location, list) and not isinstance(node_location[0], (int, float)):
            for loc in node_location:
                self.fixNode(loc, dofs)
            return
        
        # Determine if node_location is ID or position
        if isinstance(node_location, (int, np.integer)):
            node_id = node_location
        else:
            node_id = self.find_node(node_location)
            if node_id is None:
                raise ValueError(f"No node found at {node_location}")
        
        self.fix_node_by_id(node_id, dofs)
    
    # =========================================================================
    # ABSTRACT METHODS (Children must implement)
    # =========================================================================
    
    @abstractmethod
    def get_K_str(self):
        """
        Assemble global stiffness matrix.
        
        Children must implement this to build self.K from elements.
        """
        pass
    
    @abstractmethod
    def get_P_r(self):
        """
        Assemble residual force vector.
        
        Children must implement this to build self.P_r from internal forces.
        """
        pass
    
    # =========================================================================
    # SOLVING (Concrete - children can override)
    # =========================================================================
    
    def solve_linear(self) -> np.ndarray:
        """
        Solve linear system K*U = P.
        
        Returns:
            U: Displacement vector
            
        Example:
            structure.finalize()
            U = structure.solve_linear()
            print(f"Max displacement: {np.max(np.abs(U))}")
        """
        from scipy.sparse.linalg import spsolve
        from scipy.sparse import csr_matrix
        
        if not self._finalized:
            raise RuntimeError("Must call finalize() before solving")
        
        print("Assembling system...")
        
        # Assemble system
        self.get_K_str()
        self.get_P_r()
        
        print(f"System size: {self.nb_dofs} DOFs ({self.nb_dof_free} free)")
        
        # Extract free system
        K_free = self.K[np.ix_(self.dof_free, self.dof_free)]
        P_free = self.P_r[self.dof_free]
        
        # Convert to sparse for efficiency
        K_sparse = csr_matrix(K_free)
        
        print("Solving system...")
        U_free = spsolve(K_sparse, P_free)
        
        # Place into full displacement vector
        self.U[self.dof_free] = U_free
        
        print("Solution complete.")
        
        return self.U
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"{self.__class__.__name__}("
                f"nodes={len(self._nodes)}, "
                f"elements={len(self._elements)}, "
                f"finalized={self._finalized})")
