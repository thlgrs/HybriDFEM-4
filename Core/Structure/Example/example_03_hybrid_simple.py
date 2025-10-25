"""
Example 03: Hybrid FEM-DEM structure

This example demonstrates coupling FEM and rigid blocks.

Author: HybriDFEM Team
"""

import numpy as np
from hybridfem import Build_Hybrid


def example_hybrid_simple():
    """
    Create simple hybrid structure: masonry wall on FEM foundation.
    
    Structure:
        - FEM foundation (elastic continuum)
        - Masonry wall on top (rigid blocks)
        - Coupled at interface
    """
    print("="*70)
    print("EXAMPLE 03: HYBRID FEM-DEM STRUCTURE")
    print("="*70)
    
    # Create hybrid builder
    bh = Build_Hybrid(coupling_strategy='penalty')
    
    # =========================================================================
    # ADD FEM FOUNDATION
    # =========================================================================
    print("\nAdding FEM foundation...")
    
    bh.add_plate(
        origin=[0.0, -0.3],  # Below grade
        Lx=3.0,
        Ly=0.3,
        nx=6,  # 6 elements in x
        ny=2,  # 2 elements in y
        E=30e9,
        nu=0.2,
        thickness=0.2
    )
    
    print("  FEM foundation geometry added")
    
    # =========================================================================
    # ADD MASONRY WALL
    # =========================================================================
    print("\nAdding masonry wall...")
    
    bh.add_wall(
        origin=[0.0, 0.0],
        L=3.0,
        H=2.0,
        pattern='running_bond',
        rho=2400,
        b=0.2
    )
    
    print("  Masonry wall geometry added")
    
    # =========================================================================
    # CONVERT GEOMETRY
    # =========================================================================
    print("\nConverting geometry to nodes...")
    
    bh.make_nodes()
    
    # =========================================================================
    # DEFINE COUPLING (Explicit)
    # =========================================================================
    print("\nDefining FEM-block coupling...")
    
    # Find interface nodes (at y ≈ 0)
    interface_nodes = bh.find_nodes_in_region(
        xmin=0.0, xmax=3.0,
        ymin=-0.05, ymax=0.05
    )
    
    print(f"  Found {len(interface_nodes)} interface nodes")
    
    # Connect interface nodes to bottom blocks
    # TODO: Automatically find which block each node belongs to
    # For now, manually connect (simplified)
    print("  WARNING: Automatic interface detection not implemented")
    print("  Using manual connection (simplified)")
    
    # Manually connect first few interface nodes to first block
    if len(interface_nodes) > 0 and len(bh.list_blocks) > 0:
        for node_id in interface_nodes[:3]:  # Connect first 3 nodes
            bh.connect_node_to_block(node_id, block_id=0)
    
    # =========================================================================
    # BOUNDARY CONDITIONS
    # =========================================================================
    print("\nApplying boundary conditions...")
    
    # Fix foundation base
    foundation_base_nodes = bh.find_nodes_in_region(
        xmin=0.0, xmax=3.0,
        ymin=-0.4, ymax=-0.2
    )
    
    print(f"  Found {len(foundation_base_nodes)} foundation base nodes")
    
    for node_id in foundation_base_nodes:
        bh.fix_node_by_id(node_id, [0, 1, 2])
    
    print(f"  Fixed {len(foundation_base_nodes)} nodes at foundation base")
    
    # =========================================================================
    # LOADING
    # =========================================================================
    print("\nApplying loads...")
    
    # Apply load on top of wall
    top_nodes = bh.find_nodes_in_region(
        xmin=0.0, xmax=3.0,
        ymin=1.9, ymax=2.1
    )
    
    print(f"  Found {len(top_nodes)} top nodes")
    
    force_per_node = -10000.0 / max(len(top_nodes), 1)
    for node_id in top_nodes:
        bh.apply_force_to_node(node_id, dof=1, value=force_per_node)
    
    print(f"  Applied {force_per_node:.2f} N to each top node")
    
    # =========================================================================
    # SOLVE
    # =========================================================================
    print("\nSolving hybrid system...")
    
    U = bh.solve_linear()
    
    print("\nSolution complete!")
    print(f"  Max displacement: {np.max(np.abs(U)):.6e} m")
    
    # =========================================================================
    # RESULTS
    # =========================================================================
    print("\n" + "="*70)
    print("HYBRID STRUCTURE RESULTS")
    print("="*70)
    
    info = bh.get_structure_info()
    print(f"Total nodes: {info['nodes']}")
    print(f"FEM elements: {info['fem_elements']}")
    print(f"Rigid blocks: {info['blocks']}")
    print(f"Coupling connections: {info['connections']}")
    print(f"Coupling strategy: {info['coupling_strategy']}")
    print(f"Total DOFs: {info['dofs']}")
    
    # Analyze displacements
    if len(top_nodes) > 0:
        top_displacements = [U[3*nid + 1] for nid in top_nodes]
        print(f"\nTop displacement (avg): {np.mean(top_displacements):.6e} m")
    
    if len(foundation_base_nodes) > 0:
        base_displacements = [U[3*nid + 1] for nid in foundation_base_nodes]
        print(f"Foundation displacement: {np.mean(np.abs(base_displacements)):.6e} m (should be ~0)")
    
    return bh, U


def example_hybrid_with_convenience_method():
    """
    Create hybrid structure using convenience method.
    
    Demonstrates the add_masonry_wall_with_foundation() method.
    """
    print("\n\n" + "="*70)
    print("EXAMPLE 03b: HYBRID WITH CONVENIENCE METHOD")
    print("="*70)
    
    bh = Build_Hybrid()
    
    # Single convenience method creates both wall and foundation
    print("\nUsing convenience method...")
    bh.add_masonry_wall_with_foundation(
        origin=[0.0, 0.0],
        L=3.0,
        H=2.0,
        foundation_thickness=0.3,
        pattern='running_bond'
    )
    
    # Convert
    print("\nConverting geometry...")
    bh.make_nodes()
    
    # Auto-detect interfaces (not yet implemented)
    print("\nAuto-detecting interfaces...")
    try:
        bh.auto_detect_interfaces()
    except:
        print("  (auto-detection not yet implemented)")
    
    bh.print_structure_summary()
    
    return bh


if __name__ == "__main__":
    # Run simple hybrid example
    bh1, U1 = example_hybrid_simple()
    
    # Run convenience method example
    bh2 = example_hybrid_with_convenience_method()
    
    print("\n" + "="*70)
    print("Hybrid examples complete!")
    print("="*70)
