# HybriDFEM Architecture Skeleton

## Overview

This is a skeleton implementation of the refactored HybriDFEM architecture, featuring:

- **Clean separation** between low-level (monolithic) and high-level (builder) APIs
- **Multiple inheritance** from Structure_block and Structure_FEM
- **Builder patterns** for convenience (Build_blocks, Build_FEM, Build_Hybrid)
- **Backward compatibility** with deprecation warnings for legacy methods

## Architecture

```
Structure_2D (Abstract Base)
├── add_node([x, y]) → node_id
├── finalize()
├── solve_linear()
│
├─── Structure_block (Low-level blocks)
│    ├── add_rigid_block_by_nodes([n0,n1,n2,n3], ...)
│    └── add_block() [DEPRECATED]
│
├─── Structure_FEM (Low-level FEM)
│    ├── add_triangle_element([n0,n1,n2], E, nu, t)
│    ├── add_beam_element([n0,n1], E, nu, H, b)
│    └── add_fe() [DEPRECATED]
│
└─── Hybrid (Multiple Inheritance)
     ├── Inherits from Structure_block + Structure_FEM
     └── connect_node_to_block(node_id, block_id)

Build_blocks (High-level blocks)
├── Inherits from Structure_block
├── add_wall(origin, L, H, pattern)
├── add_arch(center, span, rise, n_voussoirs)
├── add_voronoi_surface(region, n_cells)
└── make_nodes()  # Converts geometry → nodes

Build_FEM (High-level FEM)
├── Inherits from Structure_FEM
├── add_plate(origin, Lx, Ly, nx, ny)
├── add_mesh(region, element_size)
└── make_nodes()  # Converts geometry → nodes

Build_Hybrid (Multiple Inheritance)
├── Inherits from Build_blocks + Build_FEM + Hybrid
├── All high-level methods from parents
├── make_nodes()  # Converts both types
└── Convenience methods (add_masonry_wall_with_foundation, etc.)
```

## File Structure

```
hybridfem/
├── __init__.py                  # Package initialization
├── structure_2d.py              # Base class (low-level)
├── structure_block.py           # Block structures (low-level)
├── structure_fem.py             # FEM structures (low-level)
├── build_blocks.py              # Block builder (high-level)
├── build_fem.py                 # FEM builder (high-level)
├── hybrid.py                    # Hybrid coupling (low-level)
├── build_hybrid.py              # Hybrid builder (high-level)
├── utils.py                     # Utility functions
└── examples/
    ├── example_01_monolithic_blocks.py
    ├── example_02_builder_wall.py
    ├── example_03_hybrid_simple.py
    └── example_04_mixed_workflow.py
```

## Usage Examples

### 1. Low-Level (Monolithic API)

For users who want **precise control** over every node and element:

```python
from hybridfem import Structure_block

# Create structure
sb = Structure_block()

# Add nodes explicitly
n0 = sb.add_node([0, 0])
n1 = sb.add_node([1, 0])
n2 = sb.add_node([1, 1])
n3 = sb.add_node([0, 1])

# Create block using nodes
block_id = sb.add_rigid_block_by_nodes(
    [n0, n1, n2, n3], 
    ref_point=[0.5, 0.5]
)

# Finalize and solve
sb.finalize()
sb.solve_linear()
```

### 2. High-Level (Builder API)

For users who want **convenience** for common patterns:

```python
from hybridfem import Build_blocks

# Create builder
bb = Build_blocks()

# Add wall with pattern
bb.add_wall([0, 0], L=3, H=2, pattern='running_bond')

# Convert geometry to nodes
bb.make_nodes()

# Apply BCs and solve
bottom_nodes = bb.find_nodes_in_region(0, 3, -0.1, 0.1)
for node_id in bottom_nodes:
    bb.fix_node_by_id(node_id, [0, 1, 2])

bb.solve_linear()
```

### 3. Hybrid FEM-DEM

Combining finite elements and rigid blocks:

```python
from hybridfem import Build_Hybrid

# Create hybrid builder
bh = Build_Hybrid()

# Add FEM foundation
bh.add_plate([0, -0.3], Lx=3, Ly=0.3, nx=6, ny=2)

# Add block wall
bh.add_wall([0, 0], L=3, H=2, pattern='running_bond')

# Convert and couple
bh.make_nodes()

# Explicit coupling
interface_nodes = bh.find_nodes_in_region(0, 3, -0.05, 0.05)
for node_id in interface_nodes:
    bh.connect_node_to_block(node_id, block_id=0)

bh.solve_linear()
```

### 4. Mixed Workflow (Power Users)

Start with builder, add custom details with monolithic API:

```python
from hybridfem import Build_Hybrid

bh = Build_Hybrid()

# Use builder for bulk
bh.add_wall([0, 0], L=3, H=2)
bh.make_nodes()

# Use monolithic API for custom additions
n_custom = bh.add_node([3.5, 1.0])
bh.add_triangle_element([n_custom, n1, n2], E=30e9, nu=0.2)

# Re-finalize and solve
bh._finalized = False
bh.finalize()
bh.solve_linear()
```

## Key Design Decisions

### 1. Two-Phase Workflow

**All classes** use a two-phase approach:

1. **Phase 1**: Add things (nodes, elements, geometry)
2. **Phase 2**: Finalize (build DOF system)
3. **Phase 3**: Solve

```python
# Phase 1: Add
structure.add_node(...)
structure.add_element(...)

# Phase 2: Finalize
structure.finalize()  # Or make_nodes() for builders

# Phase 3: Solve
structure.solve_linear()
```

### 2. Unified Node Storage

All nodes stored in `Structure_2D._nodes` (base class):
- ✅ Single source of truth
- ✅ Consistent node IDs across all classes
- ✅ No merging needed in multiple inheritance

### 3. Separate Geometry Storage

Builders store geometry separately:
- `Build_blocks._block_geometry` 
- `Build_FEM._fem_geometry`
- `Build_Hybrid` inherits both → no conflicts

### 4. Inheritance Strategy

- **Structure_block** and **Structure_FEM** inherit from **Structure_2D**
- **Hybrid** uses **multiple inheritance** from both
- **Build_blocks** inherits from **Structure_block** (IS-A relationship)
- **Build_FEM** inherits from **Structure_FEM** (IS-A relationship)
- **Build_Hybrid** inherits from **Build_blocks + Build_FEM + Hybrid**

This allows:
- Independent usage of each class
- Reuse of all parent methods
- Clear separation of concerns

## TODO List

This skeleton needs the following implementations:

### High Priority
1. [ ] **Block stiffness assembly** (`Structure_block._stiffness_block()`)
2. [ ] **FEM stiffness assembly** (`Structure_FEM._stiffness_fem()`)
3. [ ] **Coupling implementation** (`Hybrid._setup_constraint_elimination()`)
4. [ ] **Running bond wall generation** (`Build_blocks._build_running_bond_wall()`)
5. [ ] **Structured mesh generation** (`Build_FEM._build_plate()`)

### Medium Priority
6. [ ] **Arch generation** (`Build_blocks._build_arch()`)
7. [ ] **Voronoi tessellation** (`Build_blocks._build_voronoi()`)
8. [ ] **Beam mesh generation** (`Build_FEM._build_beam_mesh()`)
9. [ ] **Auto interface detection** (`Build_Hybrid.auto_detect_interfaces()`)
10. [ ] **Contact face creation** (`Structure_block.make_cfs()`)

### Low Priority
11. [ ] **Unstructured meshing** (`Build_FEM._build_mesh()`)
12. [ ] **VTK export** (`utils.export_to_vtk()`)
13. [ ] **Material classes** (separate Material objects)
14. [ ] **Nonlinear solver** (`solve_nonlinear()`)
15. [ ] **Dynamic solver** (`solve_dynamic()`)

## Implementation Notes

### Stiffness Assembly Template

```python
def _stiffness_fem(self):
    """Assemble FEM stiffness matrix."""
    for elem in self.list_fes:
        # Get element nodes
        node_ids = elem.node_ids
        positions = [self._nodes[nid] for nid in node_ids]
        
        # Compute element stiffness
        K_elem = compute_element_stiffness(positions, elem.E, elem.nu, ...)
        
        # Global DOF indices
        dofs = [3*nid + i for nid in node_ids for i in range(3)]
        
        # Assemble into global K
        self.K[np.ix_(dofs, dofs)] += K_elem
```

### Wall Generation Template

```python
def _build_running_bond_wall(self, origin, L, H, rho, b):
    """Create running bond pattern."""
    block_L, block_H = 0.4, 0.2
    n_courses = int(np.ceil(H / block_H))
    n_blocks_per_course = int(np.ceil(L / block_L))
    
    for course in range(n_courses):
        y_base = origin[1] + course * block_H
        x_offset = (block_L / 2) if (course % 2 == 1) else 0
        
        for i in range(n_blocks_per_course):
            x_left = origin[0] + i * block_L + x_offset
            x_right = min(x_left + block_L, origin[0] + L)
            y_top = y_base + block_H
            
            # Create vertices
            vertices = [
                [x_left, y_base],
                [x_right, y_base],
                [x_right, y_top],
                [x_left, y_top]
            ]
            
            # Find or create nodes
            node_ids = [self._find_or_create_node(v) for v in vertices]
            
            # Create block
            ref_point = [(x_left + x_right) / 2, (y_base + y_top) / 2]
            self.add_rigid_block_by_nodes(node_ids, ref_point, rho, b)
```

## Migration from Old API

Old code can still work with deprecation warnings:

```python
# OLD (still works)
st = Hybrid()
st.add_block(vertices, rho, b)
st.make_nodes()  # DeprecationWarning
st.solve_linear()

# NEW (recommended)
h = Hybrid()
node_ids = [h.add_node(v) for v in vertices]
h.add_rigid_block_by_nodes(node_ids, ref_point, rho, b)
h.finalize()
h.solve_linear()
```

## Testing

Run examples to verify architecture:

```bash
python examples/example_01_monolithic_blocks.py
python examples/example_02_builder_wall.py
python examples/example_03_hybrid_simple.py
python examples/example_04_mixed_workflow.py
```

## Questions?

This skeleton demonstrates the architecture. Key files to understand:

1. **structure_2d.py** - Base class with monolithic API
2. **build_blocks.py** - High-level builder pattern
3. **hybrid.py** - Multiple inheritance coupling
4. **example_04_mixed_workflow.py** - Combining approaches

---

**Next Steps**: Fill in the TODO implementations with actual mechanics from your existing HybriDFEM code!
