"""
Triangular finite elements for 2D plane stress/strain analysis.

Implements:
- T3: 3-node linear triangle (Constant Strain Triangle - CST)
- T6: 6-node quadratic triangle (Linear Strain Triangle - LST)
"""

import numpy as np
from typing import List, Tuple
from core.element import Element2D, Node


class TriangleT3(Element2D):
    """
    3-node linear triangular element (CST - Constant Strain Triangle).
    
    Node numbering (counter-clockwise):
        3
        |\\
        | \\
        |  \\
        1---2
    
    Natural coordinates: (ξ, η) where ξ, η ≥ 0 and ξ + η ≤ 1
    """
    
    def __init__(self, element_id: int, nodes: List[Node], 
                 material, thickness: float = 1.0, plane_stress: bool = True):
        """
        Initialize T3 element.
        
        Args:
            element_id: Element identifier
            nodes: List of 3 nodes
            material: Material model
            thickness: Element thickness
            plane_stress: True for plane stress, False for plane strain
        """
        if len(nodes) != 3:
            raise ValueError("T3 element requires exactly 3 nodes")
        
        super().__init__(element_id, nodes, material, thickness, plane_stress)
        
        # Precompute for efficiency (constant for linear triangle)
        self._B_matrix = None
        self._area = None
    
    def shape_functions(self, xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Linear shape functions for T3 element.
        
        N1 = 1 - ξ - η (area coordinate L1)
        N2 = ξ           (area coordinate L2)
        N3 = η           (area coordinate L3)
        
        Args:
            xi, eta: Natural coordinates
            
        Returns:
            N: Shape function values [N1, N2, N3]
            dN_dxi: Derivatives w.r.t. xi [-1, 1, 0]
            dN_deta: Derivatives w.r.t. eta [-1, 0, 1]
        """
        N = np.array([1 - xi - eta, xi, eta])
        dN_dxi = np.array([-1.0, 1.0, 0.0])
        dN_deta = np.array([-1.0, 0.0, 1.0])
        
        return N, dN_dxi, dN_deta
    
    def gauss_quadrature(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Single-point quadrature at centroid (exact for linear elements).
        
        Returns:
            xi_pts, eta_pts, weights: Quadrature data
        """
        # Centroid of reference triangle
        xi_pts = np.array([1/3])
        eta_pts = np.array([1/3])
        weights = np.array([0.5])  # Area of reference triangle
        
        return xi_pts, eta_pts, weights
    
    def compute_area(self) -> float:
        """
        Compute triangle area using coordinates.
        
        Returns:
            area: Triangle area
        """
        if self._area is not None:
            return self._area
        
        coords = self.get_node_coordinates()
        x1, y1 = coords[0]
        x2, y2 = coords[1]
        x3, y3 = coords[2]
        
        self._area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        
        if self._area <= 0:
            raise ValueError(f"Degenerate triangle in element {self.id}")
        
        return self._area
    
    def compute_B_matrix(self, xi: float = None, eta: float = None) -> Tuple[np.ndarray, float]:
        """
        Compute B matrix (constant for T3).
        
        For CST, B matrix is constant throughout element, so xi, eta are ignored.
        
        Returns:
            B: Strain-displacement matrix (3×6)
            det_J: Jacobian determinant (= 2 × area)
        """
        if self._B_matrix is not None:
            area = self.compute_area()
            return self._B_matrix, 2 * area
        
        # Use parent class method at centroid
        B, det_J = super().compute_B_matrix(1/3, 1/3)
        
        # Cache for efficiency
        self._B_matrix = B
        
        return B, det_J
    
    def compute_stiffness_matrix(self) -> np.ndarray:
        """
        Compute stiffness matrix for T3 (closed-form).
        
        K = B^T D B × t × A
        
        Returns:
            K_elem: Element stiffness matrix (6×6)
        """
        B, det_J = self.compute_B_matrix()
        area = self.compute_area()
        
        K_elem = self.thickness * area * (B.T @ self.elasticity_matrix @ B)
        
        return K_elem
    
    def compute_mass_matrix(self, lumped: bool = False) -> np.ndarray:
        """
        Compute mass matrix for T3.
        
        Args:
            lumped: If True, use lumped mass
            
        Returns:
            M_elem: Mass matrix (6×6)
        """
        rho = self.material.rho
        area = self.compute_area()
        total_mass = rho * self.thickness * area
        
        if lumped:
            # Lumped mass: divide equally among nodes
            node_mass = total_mass / 3.0
            M_elem = np.diag([node_mass, node_mass] * 3)
        else:
            # Consistent mass matrix for T3
            # ∫∫ Ni Nj dA = A/12 for i=j, A/24 for i≠j
            m_diag = total_mass / 6.0
            m_off = total_mass / 12.0
            
            M_scalar = np.array([
                [2, 1, 1],
                [1, 2, 1],
                [1, 1, 2]
            ]) * (area / 12.0) * rho * self.thickness
            
            # Expand to include both u and v DOFs
            M_elem = np.zeros((6, 6))
            M_elem[0::2, 0::2] = M_scalar  # u-u coupling
            M_elem[1::2, 1::2] = M_scalar  # v-v coupling
        
        return M_elem
    
    def get_edges(self) -> List[Tuple[int, int]]:
        """
        Get edges of triangle.
        
        Returns:
            edges: [(node0, node1), (node1, node2), (node2, node0)]
        """
        return [
            (self.nodes[0].id, self.nodes[1].id),
            (self.nodes[1].id, self.nodes[2].id),
            (self.nodes[2].id, self.nodes[0].id)
        ]


class TriangleT6(Element2D):
    """
    6-node quadratic triangular element (LST - Linear Strain Triangle).
    
    Node numbering (counter-clockwise):
        3
        |\\
        6  5
        |   \\
        1--4--2
    
    Nodes 1-3 are corners, nodes 4-6 are mid-side.
    """
    
    def __init__(self, element_id: int, nodes: List[Node],
                 material, thickness: float = 1.0, plane_stress: bool = True):
        """
        Initialize T6 element.
        
        Args:
            element_id: Element identifier
            nodes: List of 6 nodes (corners first, then mid-sides)
            material: Material model
            thickness: Element thickness
            plane_stress: True for plane stress, False for plane strain
        """
        if len(nodes) != 6:
            raise ValueError("T6 element requires exactly 6 nodes")
        
        super().__init__(element_id, nodes, material, thickness, plane_stress)
    
    def shape_functions(self, xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Quadratic shape functions for T6 element.
        
        Uses area coordinates L1 = 1-ξ-η, L2 = ξ, L3 = η
        
        Args:
            xi, eta: Natural coordinates
            
        Returns:
            N: Shape function values (6,)
            dN_dxi: Derivatives w.r.t. xi (6,)
            dN_deta: Derivatives w.r.t. eta (6,)
        """
        L1 = 1 - xi - eta
        L2 = xi
        L3 = eta
        
        # Quadratic shape functions
        N = np.array([
            L1 * (2*L1 - 1),  # N1 - corner node 1
            L2 * (2*L2 - 1),  # N2 - corner node 2
            L3 * (2*L3 - 1),  # N3 - corner node 3
            4 * L1 * L2,       # N4 - mid-side node 1-2
            4 * L2 * L3,       # N5 - mid-side node 2-3
            4 * L3 * L1        # N6 - mid-side node 3-1
        ])
        
        # Derivatives w.r.t. xi
        dN_dxi = np.array([
            4*xi + 4*eta - 3,  # dN1/dξ
            4*xi - 1,          # dN2/dξ
            0,                 # dN3/dξ
            4 - 8*xi - 4*eta,  # dN4/dξ
            4*eta,             # dN5/dξ
            -4*eta             # dN6/dξ
        ])
        
        # Derivatives w.r.t. eta
        dN_deta = np.array([
            4*xi + 4*eta - 3,  # dN1/dη
            0,                 # dN2/dη
            4*eta - 1,         # dN3/dη
            -4*xi,             # dN4/dη
            4*xi,              # dN5/dη
            4 - 4*xi - 8*eta   # dN6/dη
        ])
        
        return N, dN_dxi, dN_deta
    
    def gauss_quadrature(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        3-point Gauss quadrature (exact for quadratic elements).
        
        Returns:
            xi_pts, eta_pts, weights: Quadrature data
        """
        # 3-point rule for triangles
        xi_pts = np.array([1/6, 2/3, 1/6])
        eta_pts = np.array([1/6, 1/6, 2/3])
        weights = np.array([1/6, 1/6, 1/6])
        
        return xi_pts, eta_pts, weights
    
    def get_edges(self) -> List[Tuple[int, int, int]]:
        """
        Get edges with mid-side nodes.
        
        Returns:
            edges: [(corner1, corner2, mid_node), ...] for each edge
        """
        return [
            (self.nodes[0].id, self.nodes[1].id, self.nodes[3].id),  # Edge 1-2
            (self.nodes[1].id, self.nodes[2].id, self.nodes[4].id),  # Edge 2-3
            (self.nodes[2].id, self.nodes[0].id, self.nodes[5].id)   # Edge 3-1
        ]


# Convenience factory function
def create_triangle(element_id: int, nodes: List[Node], material,
                   thickness: float = 1.0, plane_stress: bool = True,
                   element_type: str = 'T3') -> Element2D:
    """
    Factory function to create triangular elements.
    
    Args:
        element_id: Element identifier
        nodes: List of nodes (3 for T3, 6 for T6)
        material: Material model
        thickness: Element thickness
        plane_stress: True for plane stress, False for plane strain
        element_type: 'T3' or 'T6'
        
    Returns:
        element: Triangular element instance
    """
    if element_type.upper() == 'T3':
        return TriangleT3(element_id, nodes, material, thickness, plane_stress)
    elif element_type.upper() == 'T6':
        return TriangleT6(element_id, nodes, material, thickness, plane_stress)
    else:
        raise ValueError(f"Unknown triangle type: {element_type}")
