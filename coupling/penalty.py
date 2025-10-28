"""
Coupling strategies for connecting blocks with FEM elements.

Implements different methods:
- Penalty method
- Lagrange multipliers
- Mortar method
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
import numpy as np


class CouplingStrategy(ABC):
    """
    Abstract base class for Block-FEM coupling methods.
    
    Each coupling method must implement:
    - Contact detection
    - Constraint enforcement
    - Stiffness contribution
    """
    
    def __init__(self, name: str = "Coupling"):
        self.name = name
        self.contact_pairs = []  # List of detected contacts
    
    @abstractmethod
    def detect_contacts(self, blocks: List, elements: List) -> List:
        """
        Detect potential contacts between blocks and FEM elements.
        
        Args:
            blocks: List of Block objects
            elements: List of Element2D objects
            
        Returns:
            contact_pairs: List of detected contact pairs
        """
        pass
    
    @abstractmethod
    def compute_coupling_stiffness(self) -> np.ndarray:
        """
        Compute contribution to global stiffness matrix from coupling.
        
        Returns:
            K_coupling: Coupling stiffness contribution
        """
        pass
    
    @abstractmethod
    def compute_coupling_forces(self, U: np.ndarray) -> np.ndarray:
        """
        Compute coupling forces for given displacement field.
        
        Args:
            U: Global displacement vector
            
        Returns:
            F_coupling: Coupling force vector
        """
        pass


class PenaltyCoupling(CouplingStrategy):
    """
    Penalty method for Block-FEM coupling.
    
    Enforces compatibility through stiff springs:
        F_penalty = k_penalty × gap
    
    Advantages:
    - Simple to implement
    - No additional DOFs
    - Works with standard solvers
    
    Disadvantages:
    - Requires tuning penalty parameter
    - Approximate constraint enforcement
    - May cause ill-conditioning
    """
    
    def __init__(self, k_penalty: float = 1e10, tolerance: float = 1e-3):
        """
        Initialize penalty coupling.
        
        Args:
            k_penalty: Penalty stiffness parameter
            tolerance: Proximity tolerance for contact detection
        """
        super().__init__("Penalty")
        self.k_penalty = k_penalty
        self.tolerance = tolerance
    
    def detect_contacts(self, blocks: List, elements: List) -> List:
        """
        Detect block edges near FEM element edges.
        
        Algorithm:
        1. For each block edge
        2. For each FEM element edge
        3. Check if edges are:
            - Parallel (tangent vectors aligned)
            - Close (distance < tolerance)
            - Overlapping (projected extent intersects)
        
        Args:
            blocks: List of blocks
            elements: List of FEM elements
            
        Returns:
            contact_pairs: List of BlockFEMContactPair
        """
        contact_pairs = []
        
        for block in blocks:
            block_edges = block.get_edges()
            
            for block_edge in block_edges:
                for element in elements:
                    element_edges = element.get_edges()
                    
                    for elem_edge in element_edges:
                        if self._check_contact(block_edge, elem_edge):
                            contact_pair = BlockFEMContactPair(
                                block, block_edge,
                                element, elem_edge,
                                self.k_penalty
                            )
                            contact_pairs.append(contact_pair)
        
        self.contact_pairs = contact_pairs
        return contact_pairs
    
    def _check_contact(self, block_edge, elem_edge) -> bool:
        """
        Check if a block edge and element edge are in contact.
        
        Args:
            block_edge: Block edge data
            elem_edge: Element edge data
            
        Returns:
            in_contact: True if edges should be coupled
        """
        # Extract edge coordinates
        block_coords = block_edge.get_coordinates()
        elem_coords = elem_edge.get_coordinates()
        
        # Check parallelism
        block_tangent = block_coords[1] - block_coords[0]
        elem_tangent = elem_coords[1] - elem_coords[0]
        
        block_tangent /= np.linalg.norm(block_tangent)
        elem_tangent /= np.linalg.norm(elem_tangent)
        
        dot_product = abs(np.dot(block_tangent, elem_tangent))
        if dot_product < 0.95:  # Not parallel enough
            return False
        
        # Check proximity
        # Compute perpendicular distance between edges
        distance = self._edge_distance(block_coords, elem_coords)
        if distance > self.tolerance:
            return False
        
        # Check overlap
        # Project both edges onto common direction and check intersection
        overlap = self._check_overlap(block_coords, elem_coords, block_tangent)
        if overlap < 0.1 * np.linalg.norm(block_coords[1] - block_coords[0]):
            return False
        
        return True
    
    def _edge_distance(self, coords1, coords2) -> float:
        """Compute perpendicular distance between two line segments."""
        # Simplified: distance from midpoint of edge1 to edge2
        mid1 = 0.5 * (coords1[0] + coords1[1])
        
        # Distance from point to line segment
        line_vec = coords2[1] - coords2[0]
        point_vec = mid1 - coords2[0]
        
        line_len = np.linalg.norm(line_vec)
        line_unit = line_vec / line_len
        
        proj_length = np.dot(point_vec, line_unit)
        proj_length = np.clip(proj_length, 0, line_len)
        
        closest_point = coords2[0] + proj_length * line_unit
        distance = np.linalg.norm(mid1 - closest_point)
        
        return distance
    
    def _check_overlap(self, coords1, coords2, direction) -> float:
        """Check overlapping extent of two edges projected onto direction."""
        # Project endpoints onto direction
        proj1 = np.array([np.dot(coords1[0], direction), np.dot(coords1[1], direction)])
        proj2 = np.array([np.dot(coords2[0], direction), np.dot(coords2[1], direction)])
        
        # Find overlap
        min1, max1 = min(proj1), max(proj1)
        min2, max2 = min(proj2), max(proj2)
        
        overlap_start = max(min1, min2)
        overlap_end = min(max1, max2)
        
        overlap_length = max(0, overlap_end - overlap_start)
        return overlap_length
    
    def compute_coupling_stiffness(self) -> Tuple[np.ndarray, List[int]]:
        """
        Assemble penalty coupling stiffness.
        
        Returns:
            K_coupling: Coupling stiffness matrix
            dof_indices: List of involved DOF indices
        """
        # This will be implemented based on contact_pairs
        # Each contact pair contributes to stiffness
        K_contributions = []
        dof_lists = []
        
        for cp in self.contact_pairs:
            K_cp, dofs_cp = cp.get_stiffness_contribution()
            K_contributions.append(K_cp)
            dof_lists.append(dofs_cp)
        
        return K_contributions, dof_lists
    
    def compute_coupling_forces(self, U: np.ndarray) -> np.ndarray:
        """
        Compute penalty forces for current displacements.
        
        F = k_penalty × gap
        
        Args:
            U: Global displacement vector
            
        Returns:
            F_coupling: Coupling force contribution
        """
        F_coupling = np.zeros_like(U)
        
        for cp in self.contact_pairs:
            gap = cp.compute_gap(U)
            force = self.k_penalty * gap
            
            dof_indices = cp.get_dof_indices()
            F_coupling[dof_indices] += force
        
        return F_coupling


class LagrangeMultiplierCoupling(CouplingStrategy):
    """
    Lagrange multiplier method for Block-FEM coupling.
    
    Enforces constraints exactly by augmenting system:
        [K   C^T] [u]   [f]
        [C    0 ] [λ] = [0]
    
    Advantages:
    - Exact constraint enforcement
    - No penalty parameter
    - λ has physical meaning (interface forces)
    
    Disadvantages:
    - Increased system size
    - Saddle point system (requires special solver)
    - Zero diagonal blocks (may cause issues)
    """
    
    def __init__(self, tolerance: float = 1e-3):
        """
        Initialize Lagrange multiplier coupling.
        
        Args:
            tolerance: Proximity tolerance for contact detection
        """
        super().__init__("Lagrange")
        self.tolerance = tolerance
        self.n_lambda = 0  # Number of Lagrange multiplier DOFs
    
    def detect_contacts(self, blocks: List, elements: List) -> List:
        """
        Detect contacts and create Lagrange multiplier DOFs.
        
        Same detection as penalty method, but creates constraint equations.
        
        Args:
            blocks: List of blocks
            elements: List of FEM elements
            
        Returns:
            contact_pairs: List of contact pairs with constraints
        """
        # Use similar detection as penalty method
        # Each contact pair creates constraint equations
        # Number of constraints = number of contact points × 2 (normal + tangential)
        
        contact_pairs = []
        # ... detection logic similar to penalty ...
        
        # Count Lagrange multiplier DOFs
        self.n_lambda = 2 * len(contact_pairs)  # 2 per contact point (n, t)
        
        return contact_pairs
    
    def compute_coupling_stiffness(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute augmented system matrices.
        
        Returns:
            K_augmented: Augmented stiffness including constraints
            C_matrix: Constraint matrix
        """
        # Build constraint matrix C
        # Each row represents one constraint equation
        # Columns correspond to structural DOFs
        
        raise NotImplementedError("Lagrange multiplier method to be implemented")
    
    def compute_coupling_forces(self, U: np.ndarray) -> np.ndarray:
        """
        Compute constraint residual.
        
        Args:
            U: Global displacement vector (including λ)
            
        Returns:
            residual: Constraint violation
        """
        raise NotImplementedError("Lagrange multiplier method to be implemented")


class BlockFEMContactPair:
    """
    Represents a contact pair between a block edge and FEM element edge.
    
    Handles:
    - Gap computation
    - Force calculation
    - Stiffness contribution
    """
    
    def __init__(self, block, block_edge, element, elem_edge, k_penalty):
        """
        Initialize contact pair.
        
        Args:
            block: Block object
            block_edge: Block edge data
            element: Element2D object
            elem_edge: Element edge data
            k_penalty: Penalty stiffness
        """
        self.block = block
        self.block_edge = block_edge
        self.element = element
        self.elem_edge = elem_edge
        self.k_penalty = k_penalty
        
        # Discretize interface into contact points
        self.contact_points = self._discretize_interface()
    
    def _discretize_interface(self, n_points: int = 5):
        """
        Create contact points along the interface.
        
        Args:
            n_points: Number of contact points
            
        Returns:
            contact_points: List of contact point data
        """
        contact_points = []
        
        for i in range(n_points):
            xi = i / (n_points - 1)  # 0 to 1 along edge
            
            # Position on block edge
            block_pos = self._evaluate_block_edge(xi)
            
            # Corresponding position on FEM edge
            fem_pos, fem_nodes = self._project_to_fem_edge(block_pos)
            
            contact_point = {
                'xi': xi,
                'block_pos': block_pos,
                'fem_pos': fem_pos,
                'fem_nodes': fem_nodes,
                'block_dofs': self._get_block_dofs(xi),
                'fem_dofs': self._get_fem_dofs(fem_nodes)
            }
            
            contact_points.append(contact_point)
        
        return contact_points
    
    def compute_gap(self, U: np.ndarray) -> np.ndarray:
        """
        Compute gap at all contact points.
        
        Args:
            U: Global displacement vector
            
        Returns:
            gaps: Gap values at contact points
        """
        gaps = []
        
        for cp in self.contact_points:
            # Get displacements
            u_block = U[cp['block_dofs']]
            u_fem = U[cp['fem_dofs']]
            
            # Deformed positions
            pos_block_def = cp['block_pos'] + self._transform_block_displacement(u_block, cp['xi'])
            pos_fem_def = cp['fem_pos'] + u_fem  # Simplified
            
            # Gap vector
            gap_vec = pos_block_def - pos_fem_def
            gaps.append(gap_vec)
        
        return np.array(gaps)
    
    def get_stiffness_contribution(self) -> Tuple[np.ndarray, List[int]]:
        """
        Get stiffness contribution from this contact pair.
        
        Returns:
            K_local: Local stiffness matrix
            dof_indices: Global DOF indices
        """
        # Assemble penalty stiffness
        n_dofs_total = len(self.contact_points) * 4  # 2 DOFs per point (block + FEM)
        K_local = np.zeros((n_dofs_total, n_dofs_total))
        
        for i, cp in enumerate(self.contact_points):
            # Simple penalty spring in normal direction
            k = self.k_penalty
            
            # Add to local matrix
            # ... (details depend on coordinate transformations)
        
        # Get global DOF indices
        dof_indices = []
        for cp in self.contact_points:
            dof_indices.extend(cp['block_dofs'])
            dof_indices.extend(cp['fem_dofs'])
        
        return K_local, dof_indices
    
    def _evaluate_block_edge(self, xi: float) -> np.ndarray:
        """Evaluate position on block edge at parameter xi ∈ [0,1]."""
        # Linear interpolation between edge endpoints
        coords = self.block_edge.get_coordinates()
        return (1 - xi) * coords[0] + xi * coords[1]
    
    def _project_to_fem_edge(self, point: np.ndarray) -> Tuple[np.ndarray, List]:
        """Project point onto FEM edge and find shape function weights."""
        # Find closest point on FEM edge
        # Return position and list of (node_id, weight) pairs
        raise NotImplementedError("To be implemented")
    
    def _transform_block_displacement(self, u_block: np.ndarray, xi: float) -> np.ndarray:
        """Transform block rigid body displacement to point on edge."""
        # u_block = [u_x, u_y, θ]
        # Need to account for rotation and position on edge
        raise NotImplementedError("To be implemented")
    
    def _get_block_dofs(self, xi: float) -> List[int]:
        """Get block DOF indices."""
        return self.block.dof_indices
    
    def _get_fem_dofs(self, nodes: List) -> List[int]:
        """Get FEM node DOF indices."""
        dofs = []
        for node, weight in nodes:
            dofs.extend(node.dof_indices)
        return dofs
