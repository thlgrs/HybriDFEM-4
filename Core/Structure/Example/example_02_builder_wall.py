"""
Example 02: High-level builder API for masonry wall

This example demonstrates the convenience of the builder API
for creating common structural patterns.

Author: HybriDFEM Team
"""

import numpy as np
from hybridfem import Build_blocks


def example_builder_wall():
    """
    Create masonry wall using high-level builder API.
    
    Structure:
        Single masonry wall with running bond pattern.
    """
    print("="*70)
    print("EXAMPLE 02: BUILDER API - MASONRY WALL")
    print("="*70)
    
    # Create builder
    bb = Build_blocks()
    
    # =========================================================================
    # ADD GEOMETRY (High-level)
    # =========================================================================
    print("\nAdding wall geometry...")
    
    # Add wall with running bond pattern
    bb.add_wall(
        origin=[0.0, 0.0],
        L=3.0,  # 3 meters long
        H=2.0,  # 2 meters high
        pattern='running_bond',
        rho=2400,
        b=0.2
    )
    
    print("  Wall geometry stored (not yet converted to nodes)")
    
    # =========================================================================
    # CONVERT GEOMETRY TO NODES
    # =========================================================================
    print("\nConverting geometry to nodes...")
    
    # This is where the magic happens: geometry → nodes and blocks
    bb.make_nodes()
    
    # =========================================================================
    # BOUNDARY CONDITIONS (using precise control)
    # =========================================================================
    print("\nApplying boundary conditions...")
    
    # Find bottom nodes (using query methods)
    bottom_nodes = bb.find_nodes_in_region(
        xmin=0.0, xmax=3.0,
        ymin=-0.1, ymax=0.1
    )
    
    print(f"  Found {len(bottom_nodes)} bottom nodes")
    
    # Fix all bottom nodes
    for node_id in bottom_nodes:
        bb.fix_node_by_id(node_id, [0, 1, 2])
    
    print(f"  Fixed {len(bottom_nodes)} nodes at base")
    
    # =========================================================================
    # LOADING (using precise control)
    # =========================================================================
    print("\nApplying loads...")
    
    # Find top nodes
    top_nodes = bb.find_nodes_in_region(
        xmin=0.0, xmax=3.0,
        ymin=1.9, ymax=2.1
    )
    
    print(f"  Found {len(top_nodes)} top nodes")
    
    # Apply distributed load on top
    force_per_node = -5000.0 / len(top_nodes)  # Total 5 kN
    for node_id in top_nodes:
        bb.apply_force_to_node(node_id, dof=1, value=force_per_node)
    
    print(f"  Applied {force_per_node:.2f} N to each top node")
    
    # =========================================================================
    # SOLVE
    # =========================================================================
    print("\nSolving...")
    
    U = bb.solve_linear()
    
    print("\nSolution complete!")
    print(f"  Max displacement: {np.max(np.abs(U)):.6e} m")
    
    # =========================================================================
    # RESULTS
    # =========================================================================
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Total nodes: {bb.get_node_count()}")
    print(f"Total blocks: {bb.get_block_count()}")
    print(f"Total DOFs: {bb.nb_dofs}")
    
    # Calculate average top displacement
    top_displacements = []
    for node_id in top_nodes:
        v = U[3 * node_id + 1]  # Vertical displacement
        top_displacements.append(v)
    
    avg_top_disp = np.mean(top_displacements)
    print(f"\nAverage top displacement: {avg_top_disp:.6e} m")
    print(f"Max top displacement: {np.max(np.abs(top_displacements)):.6e} m")
    
    return bb, U


def example_builder_wall_with_arch():
    """
    Create wall with arch opening using builder API.
    
    Demonstrates combining multiple geometric patterns.
    """
    print("\n\n" + "="*70)
    print("EXAMPLE 02b: BUILDER API - WALL WITH ARCH")
    print("="*70)
    
    bb = Build_blocks()
    
    # Left pier
    print("\nAdding left pier...")
    bb.add_wall(
        origin=[0.0, 0.0],
        L=0.6,
        H=2.0,
        pattern='stack_bond'
    )
    
    # Right pier
    print("Adding right pier...")
    bb.add_wall(
        origin=[2.4, 0.0],
        L=0.6,
        H=2.0,
        pattern='stack_bond'
    )
    
    # Arch
    print("Adding arch...")
    bb.add_arch(
        center=[1.5, 2.0],
        span=1.8,
        rise=0.9,
        n_voussoirs=9,
        thickness=0.3
    )
    
    # Convert to nodes
    print("\nConverting geometry...")
    bb.make_nodes()
    
    print(f"\nStructure created:")
    print(f"  Nodes: {bb.get_node_count()}")
    print(f"  Blocks: {bb.get_block_count()}")
    
    return bb


if __name__ == "__main__":
    # Run simple wall example
    bb1, U1 = example_builder_wall()
    
    # Run wall with arch example
    bb2 = example_builder_wall_with_arch()
    
    print("\n" + "="*70)
    print("Examples complete!")
    print("="*70)
    
    # Note: Visualization would go here
    # import matplotlib.pyplot as plt
    # from hybridfem.utils import plot_structure_2d
    # 
    # fig, ax = plot_structure_2d(bb1)
    # plt.show()
