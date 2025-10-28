# HybridFEM-4 Architecture Design

## 1. Architecture Overview

HybridFEM-4 follows a **modular, component-based architecture** with clear separation of concerns. The design emphasizes composition over inheritance, dependency injection, and interface-based programming.

```
┌─────────────────────────────────────────────────────────────┐
│                         GUI Layer                            │
│  ┌──────────┐ ┌───────────┐ ┌──────────┐ ┌─────────────┐  │
│  │  Editor  │ │ Visualizer│ │ Controls │ │   Reports   │  │
│  └──────────┘ └───────────┘ └──────────┘ └─────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐    │
│  │   Project    │ │   Session    │ │   Workflow      │    │
│  │   Manager    │ │   Manager    │ │   Controller    │    │
│  └──────────────┘ └──────────────┘ └──────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Core Domain Layer                       │
│  ┌───────────┐ ┌──────────┐ ┌─────────┐ ┌──────────────┐  │
│  │ Structure │ │ Elements │ │ Physics │ │   Solvers    │  │
│  └───────────┘ └──────────┘ └─────────┘ └──────────────┘  │
│  ┌───────────┐ ┌──────────┐ ┌─────────┐ ┌──────────────┐  │
│  │  Contact  │ │ Coupling │ │ Meshing │ │  Boundary    │  │
│  └───────────┘ └──────────┘ └─────────┘ └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Infrastructure Layer                      │
│  ┌───────────┐ ┌──────────┐ ┌─────────┐ ┌──────────────┐  │
│  │    I/O    │ │  Export  │ │ Logging │ │ Persistence  │  │
│  └───────────┘ └──────────┘ └─────────┘ └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 2. Core Components Design

### 2.1 Structure Module

The Structure is the central aggregate that manages the complete system. It's composed of Elements and handles their interactions.

```python
# Core/Structure/structure.py

from typing import List, Dict, Optional, Protocol
from dataclasses import dataclass
import numpy as np

class IStructure(Protocol):
    """Interface for all structure types"""
    def add_element(self, element: 'IElement') -> int: ...
    def get_dofs(self) -> np.ndarray: ...
    def assemble_stiffness(self) -> np.ndarray: ...
    def assemble_mass(self) -> np.ndarray: ...

@dataclass
class Node:
    """Represents a node in the structure"""
    id: int
    position: np.ndarray  # [x, y, z]
    dofs: List[int]  # DOF indices
    constraints: Dict[str, bool] = None  # {'x': True, 'y': False, ...}
    
class Structure:
    """
    Main structure class that manages all elements and their interactions.
    Uses composition pattern instead of inheritance.
    """
    def __init__(self, dimension: int = 2):
        self.dimension = dimension
        self.nodes: Dict[int, Node] = {}
        self.elements: List['IElement'] = []
        self.contacts: List['Contact'] = []
        self.couplings: List['Coupling'] = []
        self.physics_model: Optional['PhysicsModel'] = None
        self._next_node_id = 0
        self._dof_manager = DOFManager()
        
    def add_node(self, position: np.ndarray) -> int:
        """Add a node to the structure"""
        node_id = self._next_node_id
        self.nodes[node_id] = Node(
            id=node_id,
            position=position,
            dofs=self._dof_manager.allocate_dofs(self.dimension)
        )
        self._next_node_id += 1
        return node_id
        
    def add_element(self, element: 'IElement') -> int:
        """Add an element to the structure"""
        element.validate()
        self.elements.append(element)
        return len(self.elements) - 1
        
    def add_contact(self, contact: 'Contact'):
        """Add contact between elements"""
        self.contacts.append(contact)
        
    def add_coupling(self, coupling: 'Coupling'):
        """Add coupling between different element types"""
        self.couplings.append(coupling)
        
    def assemble_system(self) -> 'SystemMatrices':
        """Assemble global system matrices"""
        K = self._assemble_stiffness()
        M = self._assemble_mass()
        C = self._assemble_damping()
        
        # Apply contact contributions
        for contact in self.contacts:
            K += contact.compute_stiffness_contribution()
            
        # Apply coupling contributions  
        for coupling in self.couplings:
            K = coupling.apply_to_stiffness(K)
            
        return SystemMatrices(K, M, C)
        
    def _assemble_stiffness(self) -> np.ndarray:
        """Assemble global stiffness matrix"""
        n_dofs = self._dof_manager.total_dofs
        K = np.zeros((n_dofs, n_dofs))
        
        for element in self.elements:
            K_elem = element.compute_stiffness()
            dofs = element.get_global_dofs()
            K[np.ix_(dofs, dofs)] += K_elem
            
        return K
```

### 2.2 Elements Module

Elements are the building blocks. They can be FEM elements or rigid blocks, all implementing the same interface.

```python
# Core/Elements/base.py

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

class IElement(ABC):
    """Interface for all element types"""
    
    @abstractmethod
    def compute_stiffness(self) -> np.ndarray:
        """Compute element stiffness matrix"""
        pass
        
    @abstractmethod
    def compute_mass(self) -> np.ndarray:
        """Compute element mass matrix"""
        pass
        
    @abstractmethod
    def get_global_dofs(self) -> List[int]:
        """Get global DOF indices"""
        pass
        
    @abstractmethod
    def validate(self) -> bool:
        """Validate element configuration"""
        pass

class Element(IElement):
    """Base class for all elements"""
    
    def __init__(self, nodes: List[int], physics_properties: 'PhysicsProperties'):
        self.nodes = nodes
        self.physics = physics_properties
        self._cached_stiffness: Optional[np.ndarray] = None
        
    def invalidate_cache(self):
        """Invalidate cached matrices when properties change"""
        self._cached_stiffness = None

# Core/Elements/fem_elements.py

class TriangleElement(Element):
    """Triangular finite element"""
    
    def __init__(self, nodes: List[int], physics: 'ElasticProperties'):
        super().__init__(nodes, physics)
        assert len(nodes) == 3, "Triangle needs 3 nodes"
        
    def compute_stiffness(self) -> np.ndarray:
        if self._cached_stiffness is not None:
            return self._cached_stiffness
            
        # Compute B matrix (strain-displacement)
        B = self._compute_B_matrix()
        
        # Get constitutive matrix from physics
        D = self.physics.get_constitutive_matrix()
        
        # Element stiffness: K = B^T * D * B * V
        K = B.T @ D @ B * self._compute_volume()
        
        self._cached_stiffness = K
        return K
        
class BeamElement(Element):
    """Timoshenko beam element"""
    
    def __init__(self, nodes: List[int], physics: 'BeamProperties'):
        super().__init__(nodes, physics)
        assert len(nodes) == 2, "Beam needs 2 nodes"
        
    def compute_stiffness(self) -> np.ndarray:
        # Beam-specific stiffness computation
        pass

# Core/Elements/rigid_blocks.py

class RigidBlock(Element):
    """Rigid block element"""
    
    def __init__(self, vertices: List[int], reference_point: np.ndarray, 
                 physics: 'RigidBodyProperties'):
        super().__init__(vertices, physics)
        self.reference_point = reference_point
        self.boundary_edges = self._compute_boundary_edges()
        
    def compute_stiffness(self) -> np.ndarray:
        # Rigid body has infinite stiffness, handled via constraints
        return self._compute_rigid_constraints()
        
    def get_edges_for_contact(self) -> List['Edge']:
        """Get edges available for contact detection"""
        return self.boundary_edges
```

### 2.3 Physics Module  

Physics properties are decoupled from elements, allowing flexible material models.

```python
# Core/Physics/properties.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

class PhysicsProperties(ABC):
    """Base interface for all physics properties"""
    
    @abstractmethod
    def get_constitutive_matrix(self) -> np.ndarray:
        """Get material constitutive matrix"""
        pass

@dataclass
class ElasticProperties(PhysicsProperties):
    """Linear elastic material properties"""
    E: float  # Young's modulus
    nu: float  # Poisson's ratio
    rho: float  # Density
    thickness: float = 1.0  # For 2D problems
    
    def get_constitutive_matrix(self) -> np.ndarray:
        """Get elasticity matrix for plane stress"""
        factor = self.E / (1 - self.nu**2)
        return factor * np.array([
            [1, self.nu, 0],
            [self.nu, 1, 0],
            [0, 0, (1 - self.nu) / 2]
        ])

@dataclass  
class PlasticProperties(ElasticProperties):
    """Elastoplastic material properties"""
    yield_stress: float
    hardening_modulus: float
    
    def get_constitutive_matrix(self) -> np.ndarray:
        # Return tangent modulus based on current state
        pass

@dataclass
class RigidBodyProperties(PhysicsProperties):
    """Rigid body properties"""
    density: float
    thickness: float = 1.0
    
    def get_constitutive_matrix(self) -> np.ndarray:
        # Rigid bodies don't have a constitutive matrix
        raise NotImplementedError("Rigid bodies use constraints")

# Core/Physics/models.py

class PhysicsModel:
    """Manages physics for the entire structure"""
    
    def __init__(self, model_type: str = "small_strain"):
        self.model_type = model_type
        self.gravity = np.array([0, -9.81, 0])
        
    def apply_body_forces(self, structure: 'Structure') -> np.ndarray:
        """Compute body force vector"""
        pass
        
    def update_configuration(self, structure: 'Structure', displacement: np.ndarray):
        """Update for large deformation if needed"""
        pass
```

### 2.4 Contact Module

Contact detection and enforcement as a separate concern.

```python
# Core/Contact/contact.py

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

class ContactDetector:
    """Detects potential contact pairs"""
    
    def detect_pairs(self, elements: List['Element']) -> List[Tuple['Element', 'Element']]:
        """Find potentially contacting element pairs"""
        # Use spatial hashing or tree structures for efficiency
        pass

class ContactModel(ABC):
    """Interface for contact models"""
    
    @abstractmethod
    def compute_forces(self, gap: float, relative_velocity: float) -> float:
        pass
        
class PenaltyContact(ContactModel):
    """Penalty method for contact"""
    
    def __init__(self, stiffness: float = 1e10):
        self.stiffness = stiffness
        
    def compute_forces(self, gap: float, relative_velocity: float) -> float:
        if gap < 0:  # Penetration
            return -self.stiffness * gap
        return 0.0

class Contact:
    """Represents a contact between two elements"""
    
    def __init__(self, element_a: 'Element', element_b: 'Element', 
                 model: ContactModel):
        self.element_a = element_a
        self.element_b = element_b
        self.model = model
        
    def compute_stiffness_contribution(self) -> np.ndarray:
        """Compute contact contribution to global stiffness"""
        pass
```

### 2.5 Coupling Module

Handles coupling between different element types (FEM-DEM coupling).

```python
# Core/Coupling/coupling.py

from abc import ABC, abstractmethod
import numpy as np

class CouplingStrategy(ABC):
    """Interface for coupling strategies"""
    
    @abstractmethod
    def compute_coupling_matrix(self, fem_element: 'Element', 
                               rigid_block: 'RigidBlock') -> np.ndarray:
        pass

class LagrangeMultiplierCoupling(CouplingStrategy):
    """Coupling via Lagrange multipliers"""
    
    def compute_coupling_matrix(self, fem_element, rigid_block):
        # Compute constraint matrix
        pass

class PenaltyCoupling(CouplingStrategy):
    """Coupling via penalty method"""
    
    def __init__(self, penalty_parameter: float = 1e10):
        self.penalty = penalty_parameter
        
    def compute_coupling_matrix(self, fem_element, rigid_block):
        # Compute penalty coupling
        pass

class Coupling:
    """Manages coupling between different element types"""
    
    def __init__(self, strategy: CouplingStrategy):
        self.strategy = strategy
        self.coupled_pairs = []
        
    def add_coupling(self, element_a: 'Element', element_b: 'Element'):
        self.coupled_pairs.append((element_a, element_b))
        
    def apply_to_stiffness(self, K: np.ndarray) -> np.ndarray:
        """Apply coupling to stiffness matrix"""
        for elem_a, elem_b in self.coupled_pairs:
            C = self.strategy.compute_coupling_matrix(elem_a, elem_b)
            # Apply coupling matrix
        return K
```

### 2.6 Solver Module

Solvers operate on structures, with a clear interface.

```python
# Core/Solvers/base.py

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

class ISolver(ABC):
    """Interface for all solver types"""
    
    @abstractmethod
    def solve(self, structure: 'Structure') -> 'Solution':
        pass

class Solution:
    """Container for solution data"""
    
    def __init__(self):
        self.displacement: Optional[np.ndarray] = None
        self.velocity: Optional[np.ndarray] = None
        self.acceleration: Optional[np.ndarray] = None
        self.stress: Optional[np.ndarray] = None
        self.strain: Optional[np.ndarray] = None
        self.time_history: list = []

# Core/Solvers/static.py

class StaticSolver(ISolver):
    """Linear static solver"""
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        
    def solve(self, structure: 'Structure') -> Solution:
        # Assemble system
        system = structure.assemble_system()
        
        # Apply boundary conditions
        K_reduced, F_reduced = self._apply_boundary_conditions(
            system.K, structure.loads, structure.constraints
        )
        
        # Solve K*u = F
        displacement = np.linalg.solve(K_reduced, F_reduced)
        
        # Create solution
        solution = Solution()
        solution.displacement = self._expand_solution(displacement)
        
        return solution

class NonlinearStaticSolver(ISolver):
    """Nonlinear static solver using Newton-Raphson"""
    
    def __init__(self, max_iterations: int = 100, tolerance: float = 1e-6):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
    def solve(self, structure: 'Structure') -> Solution:
        # Newton-Raphson iteration
        pass

# Core/Solvers/dynamic.py

class DynamicSolver(ISolver):
    """Time integration solver"""
    
    def __init__(self, time_step: float, total_time: float, 
                 method: str = "newmark"):
        self.dt = time_step
        self.total_time = total_time
        self.method = method
        
    def solve(self, structure: 'Structure') -> Solution:
        # Time stepping scheme
        pass

# Core/Solvers/modal.py

class ModalSolver(ISolver):
    """Modal/eigenvalue solver"""
    
    def __init__(self, n_modes: int = 10):
        self.n_modes = n_modes
        
    def solve(self, structure: 'Structure') -> Solution:
        # Eigenvalue problem: K*phi = omega^2*M*phi
        pass
```

## 3. Key Design Patterns and Principles

### 3.1 Dependency Injection

```python
# Example of dependency injection in action

# Create structure with injected components
structure = Structure()
structure.set_physics_model(PhysicsModel("large_strain"))
structure.set_contact_detector(ContactDetector())

# Add elements with injected properties
elastic_props = ElasticProperties(E=30e9, nu=0.2, rho=2400)
triangle = TriangleElement(nodes=[0, 1, 2], physics=elastic_props)
structure.add_element(triangle)

# Configure solver with strategy
solver = NonlinearStaticSolver()
solver.set_convergence_criterion(EnergyNormCriterion(tol=1e-6))

# Solve
solution = solver.solve(structure)
```

### 3.2 Builder Pattern for Complex Objects

```python
# Core/Builders/structure_builder.py

class StructureBuilder:
    """Builder for complex structures"""
    
    def __init__(self):
        self.structure = Structure()
        
    def add_rectangular_mesh(self, origin, dimensions, divisions):
        """Add a rectangular FEM mesh"""
        # Generate nodes and elements
        return self
        
    def add_masonry_wall(self, origin, dimensions, brick_size):
        """Add a masonry wall with rigid blocks"""
        # Generate blocks
        return self
        
    def add_foundation(self, depth, elastic_modulus):
        """Add elastic foundation"""
        return self
        
    def apply_gravity(self):
        """Apply gravity loads"""
        return self
        
    def fix_base(self):
        """Fix all nodes at y=0"""
        return self
        
    def build(self) -> Structure:
        """Return the built structure"""
        return self.structure

# Usage
builder = StructureBuilder()
structure = (builder
    .add_foundation(depth=0.5, elastic_modulus=50e9)
    .add_masonry_wall(origin=[0, 0], dimensions=[3, 2], brick_size=[0.4, 0.2])
    .apply_gravity()
    .fix_base()
    .build())
```

### 3.3 Strategy Pattern for Algorithms

```python
# Different solving strategies can be swapped

solver = StaticSolver()

# Switch to different linear solver
solver.set_linear_solver(SparseLUSolver())  # or IterativeCGSolver()

# Switch to different convergence check
solver.set_convergence(ForceNormCriterion(tol=1e-6))  # or DisplacementIncrement()
```

### 3.4 Observer Pattern for GUI Updates

```python
# Core/Events/events.py

class EventManager:
    """Manages events and observers"""
    
    def __init__(self):
        self._observers = {}
        
    def subscribe(self, event_type: str, callback):
        if event_type not in self._observers:
            self._observers[event_type] = []
        self._observers[event_type].append(callback)
        
    def notify(self, event_type: str, data=None):
        if event_type in self._observers:
            for callback in self._observers[event_type]:
                callback(data)

# Usage in solver
class NonlinearStaticSolver(ISolver):
    def __init__(self, event_manager: EventManager = None):
        self.events = event_manager or EventManager()
        
    def solve(self, structure):
        for iteration in range(self.max_iterations):
            # ... solve step ...
            
            # Notify observers
            self.events.notify("iteration_complete", {
                "iteration": iteration,
                "residual": residual,
                "displacement": u
            })
```

## 4. Testing Strategy

### 4.1 Unit Tests

```python
# tests/test_elements.py

import pytest
import numpy as np

class TestTriangleElement:
    def test_stiffness_matrix_symmetry(self):
        """Stiffness matrix should be symmetric"""
        element = TriangleElement(
            nodes=[0, 1, 2],
            physics=ElasticProperties(E=1e6, nu=0.3, rho=1000)
        )
        K = element.compute_stiffness()
        assert np.allclose(K, K.T)
        
    def test_rigid_body_motion(self):
        """Rigid body motion should produce zero strain energy"""
        # Test implementation
        pass

class TestRigidBlock:
    def test_mass_properties(self):
        """Test mass and inertia computation"""
        pass
```

### 4.2 Integration Tests

```python
# tests/integration/test_cantilever.py

def test_cantilever_deflection():
    """Test cantilever beam deflection against analytical solution"""
    structure = Structure()
    
    # Build cantilever
    builder = StructureBuilder()
    structure = builder.add_beam(length=1.0, height=0.1).fix_left_end().build()
    
    # Apply tip load
    structure.apply_point_load(node='tip', force=[0, -1000])
    
    # Solve
    solver = StaticSolver()
    solution = solver.solve(structure)
    
    # Check against analytical solution
    analytical = calculate_cantilever_deflection(L=1.0, P=1000, E=200e9, I=8.33e-6)
    assert np.isclose(solution.displacement[-1], analytical, rtol=0.01)
```

## 5. GUI Integration

### 5.1 Model-View-Presenter Pattern

```python
# GUI/Presenters/structure_presenter.py

class StructurePresenter:
    """Presenter for structure view"""
    
    def __init__(self, model: Structure, view: 'StructureView'):
        self.model = model
        self.view = view
        
        # Subscribe to model events
        model.events.subscribe('element_added', self.on_element_added)
        model.events.subscribe('solution_complete', self.on_solution_complete)
        
        # Connect view signals
        view.element_created.connect(self.create_element)
        
    def create_element(self, element_data):
        """Handle element creation from GUI"""
        element = self.element_factory.create(element_data)
        self.model.add_element(element)
        
    def on_element_added(self, element):
        """Update view when element is added"""
        self.view.add_element_to_scene(element)
```

## 6. File Structure

```
hybridfem4/
├── Core/
│   ├── __init__.py
│   ├── Structure/
│   │   ├── structure.py
│   │   ├── node.py
│   │   └── dof_manager.py
│   ├── Elements/
│   │   ├── base.py
│   │   ├── fem_elements.py
│   │   ├── rigid_blocks.py
│   │   └── constraints.py
│   ├── Physics/
│   │   ├── properties.py
│   │   ├── models.py
│   │   └── materials/
│   ├── Contact/
│   │   ├── detector.py
│   │   ├── models.py
│   │   └── algorithms.py
│   ├── Coupling/
│   │   ├── strategies.py
│   │   └── enforcement.py
│   ├── Solvers/
│   │   ├── base.py
│   │   ├── static.py
│   │   ├── dynamic.py
│   │   └── modal.py
│   ├── Builders/
│   │   └── structure_builder.py
│   └── Events/
│       └── events.py
├── GUI/
│   ├── __init__.py
│   ├── Views/
│   ├── Presenters/
│   └── Models/
├── IO/
│   ├── importers/
│   └── exporters/
├── Utils/
│   ├── geometry.py
│   └── math_helpers.py
├── Tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
└── Examples/
    └── tutorials/
```

## 7. Benefits of This Architecture

1. **Modularity**: Each component has a single responsibility
2. **Testability**: Interfaces enable easy mocking and unit testing
3. **Scalability**: New element types, solvers, or physics models can be added without modifying existing code
4. **Maintainability**: Clear separation of concerns makes code easier to understand and modify
5. **Flexibility**: Strategy and builder patterns allow runtime configuration
6. **Performance**: Caching and lazy evaluation optimize computations
7. **GUI Integration**: Event-driven architecture decouples GUI from core logic

## 8. Migration Path from HybridFEM-3

1. **Phase 1**: Create new interfaces and base classes
2. **Phase 2**: Wrap existing implementations with new interfaces
3. **Phase 3**: Gradually refactor internals
4. **Phase 4**: Deprecate old APIs
5. **Phase 5**: Remove legacy code

## 9. Example Usage

```python
# Create a hybrid structure with masonry wall on elastic foundation

from hybridfem4 import StructureBuilder, PenaltyCoupling, NonlinearStaticSolver

# Build structure
builder = StructureBuilder()
structure = (builder
    # Add elastic foundation
    .add_rectangular_mesh(
        origin=[0, -0.5], 
        dimensions=[3, 0.5],
        divisions=[30, 5],
        material=ElasticProperties(E=50e9, nu=0.2, rho=2500)
    )
    # Add masonry wall
    .add_masonry_wall(
        origin=[0, 0],
        dimensions=[3, 2],
        brick_size=[0.4, 0.2],
        material=RigidBodyProperties(density=2400)
    )
    # Add coupling at interface
    .add_interface_coupling(
        y_coordinate=0.0,
        strategy=PenaltyCoupling(penalty_parameter=1e10)
    )
    # Apply loads and constraints
    .apply_gravity()
    .fix_base()
    .build())

# Configure solver
solver = NonlinearStaticSolver(
    max_iterations=100,
    tolerance=1e-6
)

# Solve
solution = solver.solve(structure)

# Post-process
print(f"Max displacement: {np.max(solution.displacement):.3e} m")
```

This architecture provides a solid foundation for HybridFEM-4 that is:
- **Efficient**: Through caching and optimized algorithms
- **OOP**: Using interfaces, inheritance, and composition appropriately  
- **Intuitive**: Clear naming and logical organization
- **Scalable**: Easy to extend with new features
- **Testable**: Mockable interfaces and dependency injection
- **Versatile**: Supports multiple analysis types and element formulations
