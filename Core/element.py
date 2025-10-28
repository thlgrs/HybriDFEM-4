"""
Abstract base classes for finite elements.

This module defines the interface that all element types must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np


class Element(ABC):
    """
    Abstract base class for all finite elements.
    
    All element types (1D beams, 2D surface, 3D solid) inherit from this.
    """
    
    def __init__(self, element_id: int, nodes: List['Node'], material: 'Material'):
        """
        Initialize element.
        
        Args:
            element_id: Unique element identifier
            nodes: List of Node objects defining element geometry
            material: Material model
        """
        self.id = element_id
        self.nodes = nodes
        self.material = material
        self.n_nodes = len(nodes)
        self.n_dofs_per_node = self._get_dofs_per_node()
        self.n_dofs_total = self.n_nodes * self.n_dofs_per_node
        
        # State variables
        self.stress = None
        self.strain = None
    
    @abstractmethod
    def _get_dofs_per_node(self) -> int:
        """Return number of DOFs per node for this element type."""
        pass
    
    @abstractmethod
    def compute_stiffness_matrix(self) -> np.ndarray:
        """
        Compute element stiffness matrix in global coordinates.
        
        Returns:
            K_elem: Element stiffness matrix (n_dofs × n_dofs)
        """
        pass
    
    @abstractmethod
    def compute_mass_matrix(self, lumped: bool = False) -> np.ndarray:
        """
        Compute element mass matrix.
        
        Args:
            lumped: If True, return lumped mass matrix
            
        Returns:
            M_elem: Element mass matrix (n_dofs × n_dofs)
        """
        pass
    
    @abstractmethod
    def compute_internal_forces(self, u_elem: np.ndarray) -> np.ndarray:
        """
        Compute internal forces for given element displacements.
        
        Args:
            u_elem: Element displacement vector (n_dofs,)
            
        Returns:
            f_int: Internal force vector (n_dofs,)
        """
        pass
    
    @abstractmethod
    def compute_stress(self, u_elem: np.ndarray) -> np.ndarray:
        """
        Compute stress tensor from element displacements.
        
        Args:
            u_elem: Element displacement vector
            
        Returns:
            stress: Stress tensor (depends on element type)
        """
        pass
    
    def get_node_coordinates(self) -> np.ndarray:
        """
        Get coordinates of all nodes.
        
        Returns:
            coords: Array of shape (n_nodes, n_dim)
        """
        return np.array([node.coords for node in self.nodes])
    
    def get_global_dof_indices(self) -> List[int]:
        """
        Get global DOF indices for this element.
        
        Returns:
            dof_indices: List of global DOF indices
        """
        dof_indices = []
        for node in self.nodes:
            dof_indices.extend(node.dof_indices)
        return dof_indices


class Element2D(Element):
    """
    Base class for 2D surface elements (plane stress/strain).
    
    Common functionality for triangular and quadrilateral elements.
    """
    
    def __init__(self, element_id: int, nodes: List['Node'], 
                 material: 'Material', thickness: float = 1.0,
                 plane_stress: bool = True):
        """
        Initialize 2D element.
        
        Args:
            element_id: Unique identifier
            nodes: List of nodes (3, 4, 6, 8, or 9)
            material: Material model
            thickness: Element thickness
            plane_stress: True for plane stress, False for plane strain
        """
        super().__init__(element_id, nodes, material)
        self.thickness = thickness
        self.plane_stress = plane_stress
        self.elasticity_matrix = self._compute_elasticity_matrix()
    
    def _get_dofs_per_node(self) -> int:
        """2D elements have 2 DOFs per node (u_x, u_y)."""
        return 2
    
    def _compute_elasticity_matrix(self) -> np.ndarray:
        """
        Compute elasticity matrix D for plane stress or plane strain.
        
        Returns:
            D: Elasticity matrix (3×3) for [ε_xx, ε_yy, γ_xy]
        """
        E = self.material.E
        nu = self.material.nu
        
        if self.plane_stress:
            # Plane stress
            factor = E / (1 - nu**2)
            D = factor * np.array([
                [1,  nu,  0],
                [nu,  1,  0],
                [0,   0,  (1-nu)/2]
            ])
        else:
            # Plane strain
            factor = E / ((1 + nu) * (1 - 2*nu))
            D = factor * np.array([
                [1-nu,    nu,      0],
                [nu,    1-nu,      0],
                [0,       0,  (1-2*nu)/2]
            ])
        
        return D
    
    @abstractmethod
    def shape_functions(self, xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate shape functions and derivatives at natural coordinates.
        
        Args:
            xi, eta: Natural coordinates
            
        Returns:
            N: Shape function values (n_nodes,)
            dN_dxi: Derivatives w.r.t. xi (n_nodes,)
            dN_deta: Derivatives w.r.t. eta (n_nodes,)
        """
        pass
    
    @abstractmethod
    def gauss_quadrature(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get Gauss quadrature points and weights.
        
        Returns:
            xi_pts: xi coordinates of quadrature points
            eta_pts: eta coordinates of quadrature points
            weights: Integration weights
        """
        pass
    
    def compute_B_matrix(self, xi: float, eta: float) -> np.ndarray:
        """
        Compute strain-displacement matrix B at natural coordinates.
        
        B relates strains to nodal displacements: {ε} = [B]{u}
        
        Args:
            xi, eta: Natural coordinates
            
        Returns:
            B: Strain-displacement matrix (3 × n_dofs)
        """
        N, dN_dxi, dN_deta = self.shape_functions(xi, eta)
        
        # Get nodal coordinates
        coords = self.get_node_coordinates()
        
        # Compute Jacobian
        J = np.array([
            [np.dot(dN_dxi, coords[:, 0]), np.dot(dN_dxi, coords[:, 1])],
            [np.dot(dN_deta, coords[:, 0]), np.dot(dN_deta, coords[:, 1])]
        ])
        
        det_J = np.linalg.det(J)
        if det_J <= 0:
            raise ValueError(f"Negative Jacobian determinant in element {self.id}")
        
        J_inv = np.linalg.inv(J)
        
        # Derivatives in physical coordinates
        dN_dx = J_inv[0, 0] * dN_dxi + J_inv[0, 1] * dN_deta
        dN_dy = J_inv[1, 0] * dN_dxi + J_inv[1, 1] * dN_deta
        
        # Assemble B matrix
        B = np.zeros((3, self.n_dofs_total))
        for i in range(self.n_nodes):
            B[0, 2*i] = dN_dx[i]      # ∂u/∂x
            B[1, 2*i+1] = dN_dy[i]    # ∂v/∂y
            B[2, 2*i] = dN_dy[i]      # ∂u/∂y
            B[2, 2*i+1] = dN_dx[i]    # ∂v/∂x
        
        return B, det_J
    
    def compute_stiffness_matrix(self) -> np.ndarray:
        """
        Compute element stiffness matrix using Gauss quadrature.
        
        K = ∫∫ B^T D B t dA
        
        Returns:
            K_elem: Element stiffness matrix
        """
        K_elem = np.zeros((self.n_dofs_total, self.n_dofs_total))
        
        xi_pts, eta_pts, weights = self.gauss_quadrature()
        
        for xi, eta, w in zip(xi_pts, eta_pts, weights):
            B, det_J = self.compute_B_matrix(xi, eta)
            K_elem += w * self.thickness * det_J * (B.T @ self.elasticity_matrix @ B)
        
        return K_elem
    
    def compute_internal_forces(self, u_elem: np.ndarray) -> np.ndarray:
        """
        Compute internal forces: f_int = ∫∫ B^T σ t dA
        
        Args:
            u_elem: Element displacement vector
            
        Returns:
            f_int: Internal force vector
        """
        f_int = np.zeros(self.n_dofs_total)
        
        xi_pts, eta_pts, weights = self.gauss_quadrature()
        
        for xi, eta, w in zip(xi_pts, eta_pts, weights):
            B, det_J = self.compute_B_matrix(xi, eta)
            
            # Compute strain
            strain = B @ u_elem
            
            # Compute stress
            stress = self.elasticity_matrix @ strain
            
            # Add contribution
            f_int += w * self.thickness * det_J * (B.T @ stress)
        
        return f_int
    
    def compute_stress(self, u_elem: np.ndarray) -> np.ndarray:
        """
        Compute stress at element centroid.
        
        Args:
            u_elem: Element displacement vector
            
        Returns:
            stress: Stress vector [σ_xx, σ_yy, τ_xy]
        """
        # Evaluate at centroid (xi=0, eta=0 for most elements)
        B, _ = self.compute_B_matrix(0.0, 0.0)
        strain = B @ u_elem
        stress = self.elasticity_matrix @ strain
        
        return stress
    
    def compute_mass_matrix(self, lumped: bool = False) -> np.ndarray:
        """
        Compute element mass matrix.
        
        Args:
            lumped: If True, use lumped mass matrix
            
        Returns:
            M_elem: Mass matrix
        """
        if lumped:
            return self._compute_lumped_mass()
        else:
            return self._compute_consistent_mass()
    
    def _compute_consistent_mass(self) -> np.ndarray:
        """Compute consistent mass matrix."""
        M_elem = np.zeros((self.n_dofs_total, self.n_dofs_total))
        
        rho = self.material.rho
        xi_pts, eta_pts, weights = self.gauss_quadrature()
        
        for xi, eta, w in zip(xi_pts, eta_pts, weights):
            N, _, _ = self.shape_functions(xi, eta)
            _, det_J = self.compute_B_matrix(xi, eta)
            
            # Mass matrix in scalar form
            N_mat = np.zeros((2, self.n_dofs_total))
            for i in range(self.n_nodes):
                N_mat[0, 2*i] = N[i]
                N_mat[1, 2*i+1] = N[i]
            
            M_elem += w * rho * self.thickness * det_J * (N_mat.T @ N_mat)
        
        return M_elem
    
    def _compute_lumped_mass(self) -> np.ndarray:
        """Compute lumped mass matrix."""
        # Simple row-sum lumping
        M_consistent = self._compute_consistent_mass()
        M_lumped = np.diag(M_consistent.sum(axis=1))
        return M_lumped
    
    def get_edges(self) -> List[Tuple[int, int]]:
        """
        Get element edges as node index pairs.
        
        Returns:
            edges: List of (node_i, node_j) tuples
        """
        raise NotImplementedError("Subclass must implement get_edges()")


class Node:
    """
    Node class for finite element mesh.
    """
    
    def __init__(self, node_id: int, coords: np.ndarray):
        """
        Initialize node.
        
        Args:
            node_id: Unique node identifier
            coords: Coordinate array [x, y] or [x, y, z]
        """
        self.id = node_id
        self.coords = coords
        self.dof_indices = []  # Will be assigned by DOFManager
    
    def __repr__(self):
        return f"Node({self.id}, {self.coords})"
