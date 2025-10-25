"""
Structure_FEM - Low-level FEM structures

This module provides precise control over finite element creation and assembly.

Author: HybriDFEM Team
Date: 2025
"""

import numpy as np
import warnings
from typing import List, Optional
from structure_2d import Structure_2D


class Structure_FEM(Structure_2D):
    """
    Low-level finite element structures.
    
    For users who want precise control over FEM element creation using
    existing nodes.
    
    Usage:
        fem = Structure_FEM()
        n0 = fem.add_node([0, 0])
        n1 = fem.add_node([1, 0])
        n2 = fem.add_node([0, 1])
        elem_id = fem.add_triangle_element(
            [n0, n1, n2], 
            E=30e9, 
            nu=0.2, 
            thickness=1.0
        )
        fem.finalize()
        fem.solve_linear()
    """
    
    def __init__(self):
        """Initialize FEM structure."""
        super().__init__()
        
        # Legacy storage (for compatibility)
        self.list_fes = []  # Finite elements
    
    # =========================================================================
    # FEM ELEMENT CREATION (Monolithic API)
    # =========================================================================
    
    def add_triangle_element(self, node_ids: List[int], 
                            E: float, nu: float,
                            thickness: float = 1.0) -> int:
        """
        Add triangular element by node IDs (monolithic API).
        
        Parameters:
            node_ids: [n0, n1, n2] - three node IDs
            E: Young's modulus (Pa)
            nu: Poisson's ratio
            thickness: Element thickness (m)
            
        Returns:
            element_id: Integer ID of created element
            
        Example:
            fem = Structure_FEM()
            n0 = fem.add_node([0, 0])
            n1 = fem.add_node([1, 0])
            n2 = fem.add_node([0, 1])
            elem = fem.add_triangle_element([n0, n1, n2], E=30e9, nu=0.2)
        
        Note:
            Uses constant strain triangle (CST) formulation.
            Nodes should be ordered counter-clockwise.
        """
        # Validate nodes exist
        for nid in node_ids:
            if nid not in self._nodes:
                raise ValueError(f"Node {nid} doesn't exist. Add it with add_node() first.")
        
        if len(node_ids) != 3:
            raise ValueError("Triangle element requires exactly 3 nodes")
        
        # Get node positions
        positions = [self._nodes[nid] for nid in node_ids]
        
        # TODO: Create TriangleElement object
        # from your_uploaded_code import TriangleElement
        # elem = TriangleElement(positions, E=E, nu=nu, thickness=thickness)
        # elem.node_ids = node_ids
        
        # Placeholder: simple element representation
        elem = {
            'type': 'triangle',
            'node_ids': node_ids,
            'positions': positions,
            'E': E,
            'nu': nu,
            'thickness': thickness
        }
        
        elem_id = len(self.list_fes)
        self.list_fes.append(elem)
        self._elements.append(elem)  # Unified storage
        
        return elem_id
    
    def add_beam_element(self, node_ids: List[int],
                        E: float, nu: float,
                        height: float, thickness: float) -> int:
        """
        Add beam element by node IDs (monolithic API).
        
        Parameters:
            node_ids: [n1, n2] - two node IDs
            E: Young's modulus (Pa)
            nu: Poisson's ratio
            height: Beam height/depth (m)
            thickness: Beam thickness/width (m)
            
        Returns:
            element_id: Integer ID of created element
            
        Example:
            fem = Structure_FEM()
            n0 = fem.add_node([0, 0])
            n1 = fem.add_node([1, 0])
            elem = fem.add_beam_element([n0, n1], E=30e9, nu=0.2, 
                                       height=0.3, thickness=0.2)
        
        Note:
            Uses Timoshenko beam formulation.
        """
        # Validate nodes exist
        for nid in node_ids:
            if nid not in self._nodes:
                raise ValueError(f"Node {nid} doesn't exist. Add it with add_node() first.")
        
        if len(node_ids) != 2:
            raise ValueError("Beam element requires exactly 2 nodes")
        
        # Get node positions
        positions = [self._nodes[nid] for nid in node_ids]
        
        # TODO: Create BeamElement object
        # from your_existing_code import FiniteElement
        # elem = FiniteElement(positions[0], positions[1], E, nu, height, thickness)
        # elem.node_ids = node_ids
        
        # Placeholder
        elem = {
            'type': 'beam',
            'node_ids': node_ids,
            'positions': positions,
            'E': E,
            'nu': nu,
            'height': height,
            'thickness': thickness
        }
        
        elem_id = len(self.list_fes)
        self.list_fes.append(elem)
        self._elements.append(elem)
        
        return elem_id
    
    def add_quad_element(self, node_ids: List[int],
                        E: float, nu: float,
                        thickness: float = 1.0) -> int:
        """
        Add quadrilateral element by node IDs.
        
        Parameters:
            node_ids: [n0, n1, n2, n3] - four node IDs (CCW order)
            E: Young's modulus
            nu: Poisson's ratio
            thickness: Element thickness
            
        Returns:
            element_id
        """
        # Validate
        for nid in node_ids:
            if nid not in self._nodes:
                raise ValueError(f"Node {nid} doesn't exist")
        
        if len(node_ids) != 4:
            raise ValueError("Quad element requires exactly 4 nodes")
        
        # TODO: Implement quad element
        # For now, could split into two triangles
        positions = [self._nodes[nid] for nid in node_ids]
        
        elem = {
            'type': 'quad',
            'node_ids': node_ids,
            'positions': positions,
            'E': E,
            'nu': nu,
            'thickness': thickness
        }
        
        elem_id = len(self.list_fes)
        self.list_fes.append(elem)
        self._elements.append(elem)
        
        return elem_id
    
    def get_element_nodes(self, element_id: int) -> List[int]:
        """
        Get node IDs for a specific element.
        
        Parameters:
            element_id: Element ID
            
        Returns:
            List of node IDs
        """
        if element_id >= len(self.list_fes):
            raise ValueError(f"Element {element_id} doesn't exist")
        
        elem = self.list_fes[element_id]
        
        if isinstance(elem, dict):
            return elem['node_ids']
        else:
            return elem.node_ids
    
    # =========================================================================
    # LEGACY API (Deprecated)
    # =========================================================================
    
    def add_fe(self, N1: np.ndarray, N2: np.ndarray, 
               E: float, nu: float, H: float, b: float, **kwargs) -> int:
        """
        Add finite element by endpoints (LEGACY method).
        
        .. deprecated:: 4.0
            Use add_beam_element() instead for explicit control.
            
        Parameters:
            N1, N2: Endpoint positions
            E: Young's modulus
            nu: Poisson's ratio
            H: Height
            b: Thickness
            
        Returns:
            element_id
        """
        warnings.warn(
            "add_fe() is deprecated. Use add_beam_element() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Convert to monolithic API
        n1_id = self.find_node(N1, tolerance=1e-6)
        if n1_id is None:
            n1_id = self.add_node(N1)
        
        n2_id = self.find_node(N2, tolerance=1e-6)
        if n2_id is None:
            n2_id = self.add_node(N2)
        
        return self.add_beam_element([n1_id, n2_id], E, nu, H, b)
    
    # =========================================================================
    # ASSEMBLY
    # =========================================================================
    
    def _stiffness_fem(self):
        """
        Assemble FEM stiffness matrix.
        
        TODO: Implement FEM stiffness assembly
        - Loop over elements
        - Compute element stiffness matrices
        - Assemble into global K
        """
        print("  Assembling FEM stiffness...")
        
        # TODO: Implement actual stiffness assembly
        # Pseudocode:
        # for elem in self.list_fes:
        #     K_elem = elem.compute_stiffness()
        #     dofs = [3*nid+i for nid in elem.node_ids for i in range(3)]
        #     self.K[np.ix_(dofs, dofs)] += K_elem
        
        pass
    
    def get_K_str(self):
        """Assemble global stiffness matrix."""
        if not self._finalized:
            raise RuntimeError("Must call finalize() before assembly")
        
        # Initialize global stiffness
        self.K = np.zeros((self.nb_dofs, self.nb_dofs))
        
        # Assemble FEM contributions
        self._stiffness_fem()
        
        return self.K
    
    def get_P_r(self):
        """
        Assemble residual force vector.
        
        TODO: Implement residual assembly
        - Compute element internal forces
        - Assemble into global residual
        """
        if not self._finalized:
            raise RuntimeError("Must call finalize() before assembly")
        
        # TODO: Implement actual residual assembly
        # For now, residual = external forces
        self.P_r = self.P.copy()
        
        return self.P_r
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def get_element_count(self) -> int:
        """Get total number of FEM elements."""
        return len(self.list_fes)
