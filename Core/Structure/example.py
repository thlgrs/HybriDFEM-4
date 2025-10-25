"""
Example 04: Mixed workflow (builder + monolithic)

This example demonstrates how power users can combine
high-level convenience with low-level precision.

Author: HybriDFEM Team
"""

import numpy as np
from hybridfem import Build_Hybrid


def example_mixed_workflow():
    """
    Demonstrate mixed workflow: builder for bulk, monolithic for details.
    
    Strategy:
        1. Use builder for regular structure
        2. Use make_nodes() to convert
        3. Use monolithic API to add custom details
        4. Solve
    """
    print("="*70)
    print("EXAMPLE 04: MIXED WORKFLOW")
    print("="*70)
    
    # Create hybrid builder
    bh = Build_Hybrid()
    
    # =========================================================================
    # PHASE 1: Use builder for bulk structure
    # =========================================================================
    print("\nPHASE 1: Using builder API for bulk structure...")
    
    # Add main wall structure
    print("  Adding main wall...")
    bh.add_wall(
        origin=[0.0, 0.0],
        L=3.0,
        H=2.0,
        pattern='running_bond'
    )
    
    # Add foundation
    print("  Adding foundation...")
    bh.add_plate(
        origin=[0.0, -0.3],
        Lx=3.0,
        Ly=0.3,
        nx=6,
        ny=2,
        E=30e9,
        nu=0.2
    )
    
    # Convert geometry to nodes
    print("\n  Converting geometry to nodes...")
    bh.make_nodes()
    
    print(f"\n  After builder phase:")
    print(f"    Nodes: {bh.get_node_count()}")
    print(f"    Blocks: {bh.get_block_count()}")
    print(f"    FEM elements: {bh.get_element_count()}")
    
    # =========================================================================
    # PHASE 2: Use monolithic API for custom additions
    # =========================================================================
    print("\nPHASE 2: Using monolithic API for custom details...")
    
    # Example: Add a custom buttress block at precise location
    print("  Adding custom buttress block...")
    
    # Create nodes for buttress
    b0 = bh.add_node([3.0, 0.0])
    b1 = bh.add_node([3.5, 0.0])
    b2 = bh.add_node([3.3, 0.8])
    b3 = bh.add_node([3.0, 0.8])
    
    print(f"    Created nodes: {b0}, {b1}, {b2}, {b3}")
    
    # Create buttress block
    buttress_id = bh.add_rigid_block_by_nodes(
        node_ids=[b0, b1, b2, b3],
        ref_point=[3.2, 0.4],
        rho=2400,
        b=0.2
    )
    
    print(f"    Created buttress block: {buttress_id}")
    
    # Connect buttress to main wall
    # Find wall nodes near buttress interface
    wall_interface_nodes = bh.find_nodes_in_region(
        xmin=2.9, xmax=3.1,
        ymin=0.0, ymax=0.9
    )
    
    print(f"    Found {len(wall_interface_nodes)} wall interface nodes")
    
    # Manually connect (this is where precise control helps)
    if len(wall_interface_nodes) > 0:
        # Connect buttress base to first wall block
        bh.connect_node_to_block(b0, block_id=0)
        bh.connect_node_to_block(b3, block_id=0)
        print(f"    Connected buttress to wall block 0")
    
    # Add custom FEM element (triangular brace)
    print("\n  Adding custom FEM brace element...")
    
    # Find a node on wall
    wall_top_nodes = bh.find_nodes_in_region(2.9, 3.1, 1.9, 2.1)
    
    if len(wall_top_nodes) > 0:
        # Create apex node for brace
        apex = bh.add_node([3.3, 1.5])
        
        # Create triangular FEM element
        brace_elem_id = bh.add_triangle_element(
            node_ids=[wall_top_nodes[0], b2, apex],
            E=200e9,  # Steel brace
            nu=0.3,
            thickness=0.01
        )
        
        print(f"    Created brace element: {brace_elem_id}")
    
    # Need to re-finalize after adding custom elements
    print("\n  Re-finalizing structure...")
    bh._finalized = False  # Reset flag
    bh.finalize()
    
    print(f"\n  After custom additions:")
    print(f"    Nodes: {bh.get_node_count()}")
    print(f"    Blocks: {bh.get_block_count()}")
    print(f"    FEM elements: {bh.get_element_count()}")
    
    # =========================================================================
    # PHASE 3: Apply BCs and solve
    # =========================================================================
    print("\nPHASE 3: Boundary conditions and solving...")
    
    # Fix foundation base
    foundation_base = bh.find_nodes_in_region(0.0, 3.5, -0.4, -0.2)
    for node_id in foundation_base:
        bh.fix_node_by_id(node_id, [0, 1, 2])
    
    print(f"  Fixed {len(foundation_base)} foundation nodes")
    
    # Apply loads on top
    top_nodes = bh.find_nodes_in_region(0.0, 3.0, 1.9, 2.1)
    force_per_node = -10000.0 / max(len(top_nodes), 1)
    for node_id in top_nodes:
        bh.apply_force_to_node(node_id, dof=1, value=force_per_node)
    
    print(f"  Applied loads to {len(top_nodes)} top nodes")
    
    # Solve
    print("\n  Solving...")
    U = bh.solve_linear()
    
    print("\nSolution complete!")
    
    # =========================================================================
    # RESULTS
    # =========================================================================
    print("\n" + "="*70)
    print("MIXED WORKFLOW RESULTS")
    print("="*70)
    
    bh.print_structure_summary()
    
    print(f"\nMax displacement: {np.max(np.abs(U)):.6e} m")
    
    # Check buttress effectiveness
    buttress_nodes = [b0, b1, b2, b3]
    buttress_disps = [U[3*nid + 1] for nid in buttress_nodes]
    print(f"\nButtress vertical displacement: {np.mean(buttress_disps):.6e} m")
    
    # Check brace
    if 'apex' in locals():
        apex_disp = U[3*apex + 1]
        print(f"Brace apex displacement: {apex_disp:.6e} m")
    
    return bh, U


def example_iterative_refinement():
    """
    Demonstrate iterative workflow: build, analyze, refine, repeat.
    
    This is typical in research/engineering: start simple, add complexity.
    """
    print("\n\n" + "="*70)
    print("EXAMPLE 04b: ITERATIVE REFINEMENT")
    print("="*70)
    
    # =========================================================================
    # ITERATION 1: Simple wall
    # =========================================================================
    print("\nITERATION 1: Simple wall...")
    
    bh = Build_Hybrid()
    bh.add_wall([0, 0], L=3, H=2)
    bh.make_nodes()
    
    # Fix base, load top
    bottom = bh.find_nodes_in_region(0, 3, -0.1, 0.1)
    for n in bottom:
        bh.fix_node_by_id(n, [0, 1, 2])
    
    top = bh.find_nodes_in_region(0, 3, 1.9, 2.1)
    for n in top:
        bh.apply_force_to_node(n, 1, -1000/len(top))
    
    U1 = bh.solve_linear()
    max_disp_1 = np.max(np.abs(U1))
    
    print(f"  Max displacement: {max_disp_1:.6e} m")
    
    # =========================================================================
    # ITERATION 2: Add foundation (same structure, different base)
    # =========================================================================
    print("\nITERATION 2: Adding foundation for flexibility...")
    
    bh2 = Build_Hybrid()
    bh2.add_plate([0, -0.3], Lx=3, Ly=0.3, nx=6, ny=2, E=30e9, nu=0.2)
    bh2.add_wall([0, 0], L=3, H=2)
    bh2.make_nodes()
    
    # Fix foundation base
    foundation_base = bh2.find_nodes_in_region(0, 3, -0.4, -0.2)
    for n in foundation_base:
        bh2.fix_node_by_id(n, [0, 1, 2])
    
    # Same loading
    top2 = bh2.find_nodes_in_region(0, 3, 1.9, 2.1)
    for n in top2:
        bh2.apply_force_to_node(n, 1, -1000/len(top2))
    
    U2 = bh2.solve_linear()
    max_disp_2 = np.max(np.abs(U2))
    
    print(f"  Max displacement: {max_disp_2:.6e} m")
    print(f"  Change from iteration 1: {(max_disp_2/max_disp_1 - 1)*100:.2f}%")
    
    # =========================================================================
    # ITERATION 3: Add buttress if needed
    # =========================================================================
    print("\nITERATION 3: Adding buttress if displacement too large...")
    
    if max_disp_2 > 1e-3:  # If more than 1mm
        print("  Displacement large - adding buttress...")
        
        # Add buttress using monolithic API
        b0 = bh2.add_node([3.0, 0.0])
        b1 = bh2.add_node([3.5, 0.0])
        b2 = bh2.add_node([3.3, 0.8])
        b3 = bh2.add_node([3.0, 0.8])
        
        bh2.add_rigid_block_by_nodes([b0, b1, b2, b3], [3.2, 0.4])
        
        # Re-solve would go here (need to re-finalize)
        print("  (Would re-solve with buttress)")
    else:
        print("  Displacement acceptable - no refinement needed")
    
    return bh2, U2


if __name__ == "__main__":
    # Run mixed workflow example
    bh1, U1 = example_mixed_workflow()
    
    # Run iterative refinement example
    bh2, U2 = example_iterative_refinement()
    
    print("\n" + "="*70)
    print("Mixed workflow examples complete!")
    print("="*70)
    print("\nKey takeaway:")
    print("  - Use builders for speed and convenience")
    print("  - Use monolithic API for precision and custom features")
    print("  - Combine both for maximum flexibility!")
