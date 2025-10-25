"""
Example 01: Low-level monolithic API for rigid blocks

This example demonstrates precise control over block creation
using the monolithic API.

Author: HybriDFEM Team
"""

import numpy as np
from hybridfem import Structure_block


def example_monolithic_blocks():
    """
    Create simple block structure using low-level monolithic API.
    
    Structure:
        Two rigid blocks stacked vertically.
    """
    print("="*70)
    print("EXAMPLE 01: MONOLITHIC API - RIGID BLOCKS")
    print("="*70)
    
    # Create structure
    sb = Structure_block()
    
    # =========================================================================
    # BLOCK 1: Bottom block
    # =========================================================================
    print("\nCreating Block 1 (bottom)...")
    
    # Create nodes explicitly
    n0 = sb.add_node([0.0, 0.0])
    n1 = sb.add_node([1.0, 0.0])
    n2 = sb.add_node([1.0, 0.5])
    n3 = sb.add_node([0.0, 0.5])
    
    print(f"  Created nodes: {n0}, {n1}, {n2}, {n3}")
    
    # Create block using these nodes
    block1_id = sb.add_rigid_block_by_nodes(
        node_ids=[n0, n1, n2, n3],
        ref_point=[0.5, 0.25],  # Centroid
        rho=2400,
        b=0.2
    )
    
    print(f"  Created block: {block1_id}")
    
    # =========================================================================
    # BLOCK 2: Top block (shares nodes with block 1)
    # =========================================================================
    print("\nCreating Block 2 (top)...")
    
    # Reuse bottom nodes of block 1 as top of block 2 (interface)
    n4 = sb.add_node([1.0, 1.0])
    n5 = sb.add_node([0.0, 1.0])
    
    print(f"  Reusing nodes {n2}, {n3} from Block 1")
    print(f"  Created new nodes: {n4}, {n5}")
    
    # Create block (note: sharing nodes n2 and n3)
    block2_id = sb.add_rigid_block_by_nodes(
        node_ids=[n2, n4, n5, n3],  # CCW order
        ref_point=[0.5, 0.75],
        rho=2400,
        b=0.2
    )
    
    print(f"  Created block: {block2_id}")
    
    # =========================================================================
    # BOUNDARY CONDITIONS
    # =========================================================================
    print("\nApplying boundary conditions...")
    
    # Fix bottom nodes (fully constrained)
    sb.fix_node_by_id(n0, [0, 1, 2])  # Fix all DOFs
    sb.fix_node_by_id(n1, [0, 1, 2])
    
    print(f"  Fixed nodes {n0}, {n1} (all DOFs)")
    
    # =========================================================================
    # LOADING
    # =========================================================================
    print("\nApplying loads...")
    
    # Apply vertical load on top nodes
    sb.apply_force_to_node(n4, dof=1, value=-1000.0)  # Vertical
    sb.apply_force_to_node(n5, dof=1, value=-1000.0)
    
    print(f"  Applied vertical forces to nodes {n4}, {n5}")
    
    # =========================================================================
    # SOLVE
    # =========================================================================
    print("\nFinalizing and solving...")
    
    sb.finalize()
    U = sb.solve_linear()
    
    print("\nSolution complete!")
    print(f"  Max displacement: {np.max(np.abs(U)):.6e} m")
    
    # =========================================================================
    # RESULTS
    # =========================================================================
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Total nodes: {sb.get_node_count()}")
    print(f"Total blocks: {sb.get_block_count()}")
    print(f"Total DOFs: {sb.nb_dofs}")
    print(f"Free DOFs: {sb.nb_dof_free}")
    print(f"Fixed DOFs: {sb.nb_dof_fix}")
    
    # Node displacements
    print("\nNode displacements:")
    for i in range(6):
        u = U[3*i]
        v = U[3*i + 1]
        theta = U[3*i + 2]
        print(f"  Node {i}: u={u:.6e}, v={v:.6e}, θ={theta:.6e}")
    
    return sb, U


if __name__ == "__main__":
    sb, U = example_monolithic_blocks()
    
    print("\n" + "="*70)
    print("Example complete!")
    print("="*70)
