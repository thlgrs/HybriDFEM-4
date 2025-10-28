"""
HybridFEM-4 Core Implementation Skeleton
========================================
A working skeleton implementing the key architectural patterns.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Protocol, Tuple, Any
from enum import Enum
import numpy as np
import warnings


# ============================================================================
# DOMAIN EVENTS
# ============================================================================

class EventManager:
    """Simple event manager for observer pattern"""
    
    def __init__(self):
        self._observers: Dict[str, List] = {}
        
    def subscribe(self, event_type: str, callback):
        if event_type not in self._observers:
            self._observers[event_type] = []
        self._observers[event_type].append(callback)
        
    def notify(self, event_type: str, data: Any = None):
        if event_type in self._observers:
            for callback in self._observers[event_type]:
                callback(data)


# ============================================================================
# PHYSICS MODULE
# ============================================================================

class MaterialModel(Enum):
    LINEAR_ELASTIC = "linear_elastic"
    PLASTIC = "plastic"
    RIGID = "rigid"
    HYPERELASTIC = "hyperelastic"


@dataclass
class PhysicsProperties:
    """Base class for physics properties"""
    material_model: MaterialModel
    density: float
    
    def get_constitutive_matrix(self, state: Optional[Dict] = None) -> np.ndarray:
        raise NotImplementedError


@dataclass
class ElasticProperties(PhysicsProperties):
    """Linear elastic material properties"""
    E: float  # Young's modulus
    nu: float  # Poisson's ratio
    thickness: float = 1.0
    
    def __post_init__(self):
        self.material_model = MaterialModel.LINEAR_ELASTIC
        
    def get_constitutive_matrix(self, state: Optional[Dict] = None) -> np.ndarray:
        """Get plane stress elasticity matrix"""
        factor = self.E / (1 - self.nu**2) * self.thickness
        return factor * np.array([
            [1, self.nu, 0],
            [self.nu, 1, 0],
            [0, 0, (1 - self.nu) / 2]
        ])


@dataclass
class RigidBodyProperties(PhysicsProperties):
    """Properties for rigid bodies"""
    thickness: float = 1.0
    
    def __post_init__(self):
        self.material_model = MaterialModel.RIGID
        
    def get_constitutive_matrix(self, state: Optional[Dict] = None) -> np.ndarray:
        raise ValueError("Rigid bodies don't have constitutive matrices")


# ============================================================================
# NODE AND DOF MANAGEMENT
# ============================================================================

@dataclass
class Node:
    """Represents a node in the structure"""
    id: int
    position: np.ndarray
    dofs: List[int] = field(default_factory=list)
    constraints: Dict[str, bool] = field(default_factory=dict)
    
    def is_constrained(self, direction: str) -> bool:
        return self.constraints.get(direction, False)


class DOFManager:
    """Manages degree of freedom allocation and mapping"""
    
    def __init__(self, dimensions: int = 2):
        self.dimensions = dimensions
        self.dofs_per_node = 3 if dimensions == 2 else 6  # x,y,rz for 2D
        self.next_dof = 0
        self.constrained_dofs: List[int] = []
        
    def allocate_dofs(self, n_dofs: Optional[int] = None) -> List[int]:
        """Allocate DOF indices"""
        if n_dofs is None:
            n_dofs = self.dofs_per_node
            
        dofs = list(range(self.next_dof, self.next_dof + n_dofs))
        self.next_dof += n_dofs
        return dofs
        
    def add_constraint(self, dof: int):
        """Mark a DOF as constrained"""
        if dof not in self.constrained_dofs:
            self.constrained_dofs.append(dof)
            
    @property
    def free_dofs(self) -> List[int]:
        """Get list of free DOFs"""
        all_dofs = list(range(self.next_dof))
        return [d for d in all_dofs if d not in self.constrained_dofs]


# ============================================================================
# ELEMENTS MODULE
# ============================================================================

class IElement(Protocol):
    """Interface for all element types"""
    
    def compute_stiffness(self) -> np.ndarray: ...
    def compute_mass(self) -> np.ndarray: ...
    def get_global_dofs(self) -> List[int]: ...
    def get_nodes(self) -> List[int]: ...
    def update_state(self, displacement: np.ndarray): ...


@dataclass
class Element:
    """Base element class"""
    node_ids: List[int]
    physics: PhysicsProperties
    _cached_stiffness: Optional[np.ndarray] = field(default=None, init=False)
    _cached_mass: Optional[np.ndarray] = field(default=None, init=False)
    
    def invalidate_cache(self):
        self._cached_stiffness = None
        self._cached_mass = None
        
    def get_nodes(self) -> List[int]:
        return self.node_ids
        
    def update_state(self, displacement: np.ndarray):
        """Update element state for nonlinear analysis"""
        pass


class TriangleElement(Element):
    """Linear triangle finite element"""
    
    def __init__(self, node_ids: List[int], physics: ElasticProperties, 
                 node_positions: List[np.ndarray]):
        super().__init__(node_ids, physics)
        assert len(node_ids) == 3
        self.positions = node_positions
        self._compute_shape_functions()
        
    def _compute_shape_functions(self):
        """Compute shape function derivatives"""
        x = [p[0] for p in self.positions]
        y = [p[1] for p in self.positions]
        
        self.area = 0.5 * abs(
            x[0]*(y[1]-y[2]) + x[1]*(y[2]-y[0]) + x[2]*(y[0]-y[1])
        )
        
        # Shape function derivatives (constant for linear triangle)
        self.dN_dx = np.array([
            y[1] - y[2],
            y[2] - y[0], 
            y[0] - y[1]
        ]) / (2 * self.area)
        
        self.dN_dy = np.array([
            x[2] - x[1],
            x[0] - x[2],
            x[1] - x[0]
        ]) / (2 * self.area)
        
    def compute_stiffness(self) -> np.ndarray:
        """Compute element stiffness matrix"""
        if self._cached_stiffness is not None:
            return self._cached_stiffness
            
        # B matrix (strain-displacement)
        B = np.zeros((3, 6))  # 3 strains, 6 DOFs (2 per node)
        for i in range(3):
            B[0, 2*i] = self.dN_dx[i]      # ε_xx
            B[1, 2*i+1] = self.dN_dy[i]    # ε_yy
            B[2, 2*i] = self.dN_dy[i]      # γ_xy
            B[2, 2*i+1] = self.dN_dx[i]
            
        # Constitutive matrix
        D = self.physics.get_constitutive_matrix()
        
        # Element stiffness: K = B^T * D * B * Area
        K = B.T @ D @ B * self.area
        
        self._cached_stiffness = K
        return K
        
    def compute_mass(self) -> np.ndarray:
        """Compute element mass matrix"""
        if self._cached_mass is not None:
            return self._cached_mass
            
        # Lumped mass matrix
        total_mass = self.physics.density * self.area * self.physics.thickness
        node_mass = total_mass / 3
        
        M = np.zeros((6, 6))
        for i in range(3):
            M[2*i, 2*i] = node_mass      # x-direction
            M[2*i+1, 2*i+1] = node_mass  # y-direction
            
        self._cached_mass = M
        return M
        
    def get_global_dofs(self) -> List[int]:
        """Get global DOF indices for assembly"""
        # Assuming 2 DOFs per node (u, v)
        dofs = []
        for node_id in self.node_ids:
            dofs.extend([2*node_id, 2*node_id + 1])
        return dofs


class RigidBlock(Element):
    """Rigid block element"""
    
    def __init__(self, vertex_ids: List[int], physics: RigidBodyProperties,
                 vertex_positions: List[np.ndarray]):
        super().__init__(vertex_ids, physics)
        self.positions = vertex_positions
        self.compute_geometry()
        
    def compute_geometry(self):
        """Compute centroid and edges"""
        self.centroid = np.mean(self.positions, axis=0)
        
        # Compute edges for contact
        n = len(self.positions)
        self.edges = []
        for i in range(n):
            j = (i + 1) % n
            self.edges.append({
                'vertices': [self.node_ids[i], self.node_ids[j]],
                'positions': [self.positions[i], self.positions[j]],
                'normal': self._compute_edge_normal(i, j)
            })
            
    def _compute_edge_normal(self, i: int, j: int) -> np.ndarray:
        """Compute outward normal for edge"""
        edge = self.positions[j] - self.positions[i]
        normal = np.array([edge[1], -edge[0]])
        normal = normal / np.linalg.norm(normal)
        
        # Check orientation (should point outward)
        mid = 0.5 * (self.positions[i] + self.positions[j])
        if np.dot(normal, mid - self.centroid) < 0:
            normal = -normal
            
        return normal
        
    def compute_stiffness(self) -> np.ndarray:
        """Rigid blocks use constraints, not stiffness"""
        n_dofs = 2 * len(self.node_ids)
        return np.zeros((n_dofs, n_dofs))
        
    def compute_mass(self) -> np.ndarray:
        """Compute mass matrix for rigid block"""
        # Compute area (using shoelace formula)
        area = 0
        n = len(self.positions)
        for i in range(n):
            j = (i + 1) % n
            area += self.positions[i][0] * self.positions[j][1]
            area -= self.positions[j][0] * self.positions[i][1]
        area = abs(area) / 2
        
        # Total mass
        total_mass = self.physics.density * area * self.physics.thickness
        
        # Moment of inertia about centroid
        I = 0
        for pos in self.positions:
            r = np.linalg.norm(pos - self.centroid)
            I += total_mass / len(self.positions) * r**2
            
        # Rigid body has 3 DOFs: x, y, rotation
        M = np.diag([total_mass, total_mass, I])
        
        return M
        
    def get_global_dofs(self) -> List[int]:
        """Get global DOFs for rigid body (3 DOFs at centroid)"""
        # This would be mapped to actual node DOFs via constraints
        return []  # Handled through rigid constraints


# ============================================================================
# CONTACT MODULE  
# ============================================================================

class ContactModel(Enum):
    PENALTY = "penalty"
    LAGRANGE = "lagrange"
    AUGMENTED_LAGRANGE = "augmented_lagrange"


@dataclass
class ContactPair:
    """Represents a potential contact between two elements"""
    element_a: IElement
    element_b: IElement
    model: ContactModel = ContactModel.PENALTY
    penalty_stiffness: float = 1e10
    
    def compute_gap(self, positions: Dict[int, np.ndarray]) -> float:
        """Compute gap/penetration between elements"""
        # Simplified - actual implementation would be more complex
        return 0.0
        
    def compute_contact_force(self, gap: float) -> float:
        """Compute contact force based on gap"""
        if self.model == ContactModel.PENALTY:
            return max(0, -gap * self.penalty_stiffness)
        return 0.0


class ContactDetector:
    """Detects and manages contact pairs"""
    
    def __init__(self, search_radius: float = 0.1):
        self.search_radius = search_radius
        
    def detect_pairs(self, elements: List[IElement], 
                    nodes: Dict[int, Node]) -> List[ContactPair]:
        """Detect potential contact pairs"""
        pairs = []
        
        # Simple O(n²) search - real implementation would use spatial hashing
        for i, elem_a in enumerate(elements):
            for elem_b in elements[i+1:]:
                if self._are_close(elem_a, elem_b, nodes):
                    pairs.append(ContactPair(elem_a, elem_b))
                    
        return pairs
        
    def _are_close(self, elem_a: IElement, elem_b: IElement, 
                   nodes: Dict[int, Node]) -> bool:
        """Check if two elements are close enough for contact"""
        # Get element centroids
        nodes_a = elem_a.get_nodes()
        nodes_b = elem_b.get_nodes()
        
        centroid_a = np.mean([nodes[n].position for n in nodes_a], axis=0)
        centroid_b = np.mean([nodes[n].position for n in nodes_b], axis=0)
        
        return np.linalg.norm(centroid_a - centroid_b) < self.search_radius


# ============================================================================
# COUPLING MODULE
# ============================================================================

class CouplingType(Enum):
    PENALTY = "penalty"
    LAGRANGE_MULTIPLIER = "lagrange"
    MORTAR = "mortar"


@dataclass
class CouplingInterface:
    """Represents coupling between FEM and rigid blocks"""
    fem_elements: List[IElement]
    rigid_blocks: List[RigidBlock]
    coupling_type: CouplingType = CouplingType.PENALTY
    penalty_parameter: float = 1e10
    
    def compute_coupling_matrix(self) -> Tuple[np.ndarray, List[int], List[int]]:
        """Compute coupling constraint matrix"""
        # Returns C matrix and DOF indices
        pass
        
    def enforce_coupling(self, K: np.ndarray) -> np.ndarray:
        """Apply coupling to global stiffness matrix"""
        if self.coupling_type == CouplingType.PENALTY:
            # Add penalty terms
            C, fem_dofs, block_dofs = self.compute_coupling_matrix()
            K_penalty = self.penalty_parameter * C.T @ C
            # Add to appropriate locations in K
            
        return K


# ============================================================================
# STRUCTURE MODULE
# ============================================================================

@dataclass
class SystemMatrices:
    """Container for system matrices"""
    K: np.ndarray  # Stiffness
    M: Optional[np.ndarray] = None  # Mass
    C: Optional[np.ndarray] = None  # Damping
    
    
class Structure:
    """
    Main structure class - the aggregate root.
    Manages all elements and their interactions.
    """
    
    def __init__(self, dimension: int = 2):
        self.dimension = dimension
        self.nodes: Dict[int, Node] = {}
        self.elements: List[IElement] = []
        self.contact_pairs: List[ContactPair] = []
        self.coupling_interfaces: List[CouplingInterface] = []
        
        # Managers
        self.dof_manager = DOFManager(dimension)
        self.contact_detector = ContactDetector()
        self.events = EventManager()
        
        # State
        self._next_node_id = 0
        self._assembled = False
        
        # Loads and constraints
        self.point_loads: Dict[int, np.ndarray] = {}
        self.distributed_loads: List[Dict] = []
        
    def add_node(self, position: np.ndarray) -> int:
        """Add a node to the structure"""
        node_id = self._next_node_id
        node = Node(
            id=node_id,
            position=np.array(position),
            dofs=self.dof_manager.allocate_dofs()
        )
        self.nodes[node_id] = node
        self._next_node_id += 1
        
        self.events.notify("node_added", node)
        return node_id
        
    def find_node(self, position: np.ndarray, tolerance: float = 1e-6) -> Optional[int]:
        """Find existing node at position"""
        for node_id, node in self.nodes.items():
            if np.linalg.norm(node.position - position) < tolerance:
                return node_id
        return None
        
    def add_element(self, element: IElement) -> int:
        """Add an element to the structure"""
        self.elements.append(element)
        self._assembled = False
        
        self.events.notify("element_added", element)
        return len(self.elements) - 1
        
    def add_triangle(self, node_positions: List[np.ndarray], 
                    physics: ElasticProperties) -> TriangleElement:
        """Convenience method to add a triangle element"""
        # Find or create nodes
        node_ids = []
        for pos in node_positions:
            node_id = self.find_node(pos)
            if node_id is None:
                node_id = self.add_node(pos)
            node_ids.append(node_id)
            
        # Create element
        element = TriangleElement(node_ids, physics, node_positions)
        self.add_element(element)
        return element
        
    def add_rigid_block(self, vertices: List[np.ndarray], 
                       physics: RigidBodyProperties) -> RigidBlock:
        """Add a rigid block"""
        # Find or create nodes
        node_ids = []
        for pos in vertices:
            node_id = self.find_node(pos)
            if node_id is None:
                node_id = self.add_node(pos)
            node_ids.append(node_id)
            
        # Create block
        block = RigidBlock(node_ids, physics, vertices)
        self.add_element(block)
        return block
        
    def apply_constraint(self, node_id: int, directions: List[str]):
        """Apply displacement constraint to node"""
        node = self.nodes[node_id]
        for direction in directions:
            node.constraints[direction] = True
            
            # Mark DOFs as constrained
            if direction == 'x':
                self.dof_manager.add_constraint(node.dofs[0])
            elif direction == 'y':
                self.dof_manager.add_constraint(node.dofs[1])
            elif direction == 'rz' and len(node.dofs) > 2:
                self.dof_manager.add_constraint(node.dofs[2])
                
    def apply_point_load(self, node_id: int, force: np.ndarray):
        """Apply point load to node"""
        self.point_loads[node_id] = np.array(force)
        
    def detect_contacts(self):
        """Detect potential contact pairs"""
        self.contact_pairs = self.contact_detector.detect_pairs(
            self.elements, self.nodes
        )
        
    def add_coupling_interface(self, fem_nodes: List[int], 
                              block_nodes: List[int],
                              coupling_type: CouplingType = CouplingType.PENALTY):
        """Add coupling between FEM and blocks"""
        # Find elements containing these nodes
        fem_elements = []
        rigid_blocks = []
        
        for element in self.elements:
            nodes = element.get_nodes()
            if any(n in fem_nodes for n in nodes):
                fem_elements.append(element)
            if isinstance(element, RigidBlock):
                if any(n in block_nodes for n in nodes):
                    rigid_blocks.append(element)
                    
        interface = CouplingInterface(fem_elements, rigid_blocks, coupling_type)
        self.coupling_interfaces.append(interface)
        
    def assemble_system(self) -> SystemMatrices:
        """Assemble global system matrices"""
        n_dofs = self.dof_manager.next_dof
        
        # Initialize matrices
        K = np.zeros((n_dofs, n_dofs))
        M = np.zeros((n_dofs, n_dofs))
        
        # Assemble element contributions
        for element in self.elements:
            K_elem = element.compute_stiffness()
            M_elem = element.compute_mass()
            dofs = element.get_global_dofs()
            
            if len(dofs) > 0:  # Skip if no DOFs (e.g., rigid blocks)
                np.add.at(K, np.ix_(dofs, dofs), K_elem)
                np.add.at(M, np.ix_(dofs, dofs), M_elem)
                
        # Add contact contributions
        for contact in self.contact_pairs:
            # Compute contact stiffness contribution
            pass
            
        # Apply coupling
        for interface in self.coupling_interfaces:
            K = interface.enforce_coupling(K)
            
        # Compute damping (Rayleigh)
        alpha = 0.0  # Mass proportional
        beta = 0.005  # Stiffness proportional  
        C = alpha * M + beta * K
        
        self._assembled = True
        return SystemMatrices(K, M, C)
        
    def get_load_vector(self) -> np.ndarray:
        """Assemble load vector"""
        n_dofs = self.dof_manager.next_dof
        F = np.zeros(n_dofs)
        
        # Point loads
        for node_id, force in self.point_loads.items():
            node = self.nodes[node_id]
            for i, f in enumerate(force[:len(node.dofs)]):
                F[node.dofs[i]] = f
                
        # Body forces (e.g., gravity)
        # ... 
        
        return F


# ============================================================================
# SOLVERS MODULE
# ============================================================================

@dataclass
class Solution:
    """Container for analysis results"""
    displacement: np.ndarray
    velocity: Optional[np.ndarray] = None
    acceleration: Optional[np.ndarray] = None
    time: Optional[float] = None
    converged: bool = True
    iterations: int = 0
    residual_norm: float = 0.0
    

class Solver(ABC):
    """Base solver class"""
    
    def __init__(self):
        self.events = EventManager()
        
    @abstractmethod
    def solve(self, structure: Structure) -> Solution:
        pass


class StaticSolver(Solver):
    """Linear static solver"""
    
    def solve(self, structure: Structure) -> Solution:
        # Assemble system
        system = structure.assemble_system()
        K = system.K
        F = structure.get_load_vector()
        
        # Get free and fixed DOFs
        free_dofs = structure.dof_manager.free_dofs
        fixed_dofs = structure.dof_manager.constrained_dofs
        
        # Solve reduced system
        K_free = K[np.ix_(free_dofs, free_dofs)]
        F_free = F[free_dofs]
        
        # Check for singularity
        if np.linalg.cond(K_free) > 1e12:
            warnings.warn("Stiffness matrix is nearly singular")
            
        # Solve
        u_free = np.linalg.solve(K_free, F_free)
        
        # Expand solution
        u_full = np.zeros(structure.dof_manager.next_dof)
        u_full[free_dofs] = u_free
        
        # Create solution object
        solution = Solution(
            displacement=u_full,
            converged=True,
            iterations=1
        )
        
        self.events.notify("solution_complete", solution)
        return solution


class NonlinearStaticSolver(Solver):
    """Nonlinear static solver with Newton-Raphson"""
    
    def __init__(self, max_iterations: int = 50, tolerance: float = 1e-6):
        super().__init__()
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
    def solve(self, structure: Structure) -> Solution:
        n_dofs = structure.dof_manager.next_dof
        u = np.zeros(n_dofs)
        
        F_ext = structure.get_load_vector()
        free_dofs = structure.dof_manager.free_dofs
        
        for iteration in range(self.max_iterations):
            # Assemble tangent stiffness
            system = structure.assemble_system()
            K_t = system.K
            
            # Compute internal forces
            F_int = self._compute_internal_forces(structure, u)
            
            # Residual
            R = F_ext - F_int
            R_free = R[free_dofs]
            
            # Check convergence
            residual_norm = np.linalg.norm(R_free)
            
            self.events.notify("iteration_complete", {
                "iteration": iteration,
                "residual": residual_norm,
                "displacement": u
            })
            
            if residual_norm < self.tolerance:
                return Solution(
                    displacement=u,
                    converged=True,
                    iterations=iteration + 1,
                    residual_norm=residual_norm
                )
                
            # Newton update
            K_free = K_t[np.ix_(free_dofs, free_dofs)]
            du_free = np.linalg.solve(K_free, R_free)
            
            u[free_dofs] += du_free
            
            # Update element states
            for element in structure.elements:
                element.update_state(u)
                
        # Not converged
        warnings.warn(f"Newton-Raphson did not converge in {self.max_iterations} iterations")
        return Solution(
            displacement=u,
            converged=False,
            iterations=self.max_iterations,
            residual_norm=residual_norm
        )
        
    def _compute_internal_forces(self, structure: Structure, u: np.ndarray) -> np.ndarray:
        """Compute internal force vector"""
        # Simplified - actual implementation would compute from stresses
        system = structure.assemble_system()
        return system.K @ u


# ============================================================================
# BUILDER PATTERN
# ============================================================================

class StructureBuilder:
    """Builder for creating complex structures"""
    
    def __init__(self):
        self.structure = Structure()
        self._default_material = ElasticProperties(E=30e9, nu=0.2, density=2400)
        
    def set_default_material(self, material: PhysicsProperties) -> 'StructureBuilder':
        self._default_material = material
        return self
        
    def add_rectangular_fem_mesh(self, origin: np.ndarray, size: np.ndarray,
                                 divisions: Tuple[int, int],
                                 material: Optional[ElasticProperties] = None) -> 'StructureBuilder':
        """Add a rectangular FEM mesh"""
        if material is None:
            material = self._default_material
            
        dx = size[0] / divisions[0]
        dy = size[1] / divisions[1]
        
        # Create nodes
        node_grid = {}
        for j in range(divisions[1] + 1):
            for i in range(divisions[0] + 1):
                pos = origin + np.array([i * dx, j * dy])
                node_id = self.structure.add_node(pos)
                node_grid[(i, j)] = node_id
                
        # Create elements (triangles)
        for j in range(divisions[1]):
            for i in range(divisions[0]):
                # Two triangles per quad
                n1 = node_grid[(i, j)]
                n2 = node_grid[(i+1, j)]
                n3 = node_grid[(i+1, j+1)]
                n4 = node_grid[(i, j+1)]
                
                # Lower triangle
                pos1 = self.structure.nodes[n1].position
                pos2 = self.structure.nodes[n2].position
                pos3 = self.structure.nodes[n3].position
                elem1 = TriangleElement([n1, n2, n3], material, [pos1, pos2, pos3])
                self.structure.add_element(elem1)
                
                # Upper triangle
                pos4 = self.structure.nodes[n4].position
                elem2 = TriangleElement([n1, n3, n4], material, [pos1, pos3, pos4])
                self.structure.add_element(elem2)
                
        return self
        
    def add_masonry_wall(self, origin: np.ndarray, size: np.ndarray,
                        brick_size: np.ndarray,
                        material: Optional[RigidBodyProperties] = None) -> 'StructureBuilder':
        """Add a masonry wall with rigid blocks"""
        if material is None:
            material = RigidBodyProperties(density=2400)
            
        n_x = int(size[0] / brick_size[0])
        n_y = int(size[1] / brick_size[1])
        
        for j in range(n_y):
            # Offset for running bond
            offset = (brick_size[0] / 2) if j % 2 == 1 else 0
            
            for i in range(n_x):
                # Block corners
                x_left = origin[0] + i * brick_size[0] + offset
                x_right = min(x_left + brick_size[0], origin[0] + size[0])
                y_bottom = origin[1] + j * brick_size[1]
                y_top = y_bottom + brick_size[1]
                
                # Create block
                vertices = [
                    np.array([x_left, y_bottom]),
                    np.array([x_right, y_bottom]),
                    np.array([x_right, y_top]),
                    np.array([x_left, y_top])
                ]
                
                self.structure.add_rigid_block(vertices, material)
                
        return self
        
    def fix_nodes_at_y(self, y_coord: float, tolerance: float = 1e-6) -> 'StructureBuilder':
        """Fix all nodes at a given y coordinate"""
        for node_id, node in self.structure.nodes.items():
            if abs(node.position[1] - y_coord) < tolerance:
                self.structure.apply_constraint(node_id, ['x', 'y'])
                
        return self
        
    def apply_gravity(self, g: float = -9.81) -> 'StructureBuilder':
        """Apply gravity loads to all elements"""
        # Simplified - actual implementation would compute based on element mass
        for node_id, node in self.structure.nodes.items():
            # Apply some gravity load
            self.structure.apply_point_load(node_id, np.array([0, g * 100]))
            
        return self
        
    def build(self) -> Structure:
        """Return the built structure"""
        return self.structure


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_hybrid_structure():
    """Example: Masonry wall on elastic foundation"""
    
    # Create structure using builder
    builder = StructureBuilder()
    
    # Set materials
    foundation_material = ElasticProperties(E=50e9, nu=0.2, density=2500)
    masonry_material = RigidBodyProperties(density=2400)
    
    # Build structure
    structure = (builder
        # Add elastic foundation
        .add_rectangular_fem_mesh(
            origin=np.array([0, -0.5]),
            size=np.array([3.0, 0.5]),
            divisions=(10, 5),
            material=foundation_material
        )
        # Add masonry wall
        .add_masonry_wall(
            origin=np.array([0, 0]),
            size=np.array([3.0, 2.0]),
            brick_size=np.array([0.4, 0.2]),
            material=masonry_material
        )
        # Apply constraints
        .fix_nodes_at_y(y_coord=-0.5)
        # Apply loads
        .apply_gravity()
        # Build
        .build())
    
    # Add coupling at interface (y = 0)
    interface_fem_nodes = []
    interface_block_nodes = []
    
    for node_id, node in structure.nodes.items():
        if abs(node.position[1]) < 0.01:
            # Check if node belongs to FEM or block
            # (simplified logic)
            if node.position[1] < 0:
                interface_fem_nodes.append(node_id)
            else:
                interface_block_nodes.append(node_id)
                
    structure.add_coupling_interface(
        interface_fem_nodes,
        interface_block_nodes,
        CouplingType.PENALTY
    )
    
    # Solve
    solver = StaticSolver()
    solution = solver.solve(structure)
    
    print(f"Analysis complete!")
    print(f"Max displacement: {np.max(np.abs(solution.displacement)):.3e} m")
    print(f"Number of DOFs: {structure.dof_manager.next_dof}")
    print(f"Number of elements: {len(structure.elements)}")
    
    return structure, solution


if __name__ == "__main__":
    # Run example
    structure, solution = example_hybrid_structure()
