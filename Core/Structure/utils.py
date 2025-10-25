"""
Utilities for HybriDFEM

Common utility functions for geometry, visualization, etc.

Author: HybriDFEM Team
Date: 2025
"""

import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt


def compute_centroid(vertices: np.ndarray) -> np.ndarray:
    """
    Compute centroid of polygon.
    
    Parameters:
        vertices: Nx2 array of vertex coordinates
        
    Returns:
        [x, y] centroid position
    """
    return np.mean(vertices, axis=0)


def compute_polygon_area(vertices: np.ndarray) -> float:
    """
    Compute area of polygon using shoelace formula.
    
    Parameters:
        vertices: Nx2 array of vertex coordinates (CCW order)
        
    Returns:
        Area (positive for CCW, negative for CW)
    """
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """
    Test if point is inside polygon (ray casting).
    
    Parameters:
        point: [x, y] point to test
        polygon: Nx2 array of polygon vertices
        
    Returns:
        True if point inside polygon
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def rotate_points(points: np.ndarray, angle: float, 
                 center: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Rotate points around center.
    
    Parameters:
        points: Nx2 array of points
        angle: Rotation angle (radians)
        center: [x, y] rotation center (default: origin)
        
    Returns:
        Rotated points
    """
    if center is None:
        center = np.array([0, 0])
    
    # Rotation matrix
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    
    # Translate to origin, rotate, translate back
    translated = points - center
    rotated = translated @ R.T
    return rotated + center


def plot_structure_2d(structure, show_nodes: bool = True,
                     show_elements: bool = True,
                     show_blocks: bool = True,
                     node_size: float = 20,
                     figsize: Tuple[float, float] = (10, 8)):
    """
    Plot 2D structure.
    
    Parameters:
        structure: Structure_2D instance (or child)
        show_nodes: Show node markers
        show_elements: Show FEM elements
        show_blocks: Show rigid blocks
        node_size: Node marker size
        figsize: Figure size
        
    Returns:
        fig, ax: Matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get all nodes
    nodes = structure.get_all_nodes()
    
    # Plot nodes
    if show_nodes and len(nodes) > 0:
        positions = np.array([pos for pos in nodes.values()])
        ax.scatter(positions[:, 0], positions[:, 1], 
                  s=node_size, c='black', zorder=10, label='Nodes')
    
    # Plot FEM elements
    if show_elements and hasattr(structure, 'list_fes'):
        for elem in structure.list_fes:
            if isinstance(elem, dict):
                node_ids = elem['node_ids']
            else:
                node_ids = elem.node_ids
            
            positions = np.array([structure._nodes[nid] for nid in node_ids])
            
            # Close the polygon for visualization
            positions = np.vstack([positions, positions[0]])
            
            ax.plot(positions[:, 0], positions[:, 1], 
                   'b-', linewidth=1, alpha=0.5)
    
    # Plot blocks
    if show_blocks and hasattr(structure, 'list_blocks'):
        for block in structure.list_blocks:
            if isinstance(block, dict):
                vertices = block['vertices']
            else:
                vertices = block.v  # Assuming block has vertices attribute
            
            # Close the polygon
            vertices_closed = np.vstack([vertices, vertices[0]])
            
            ax.fill(vertices_closed[:, 0], vertices_closed[:, 1],
                   color='gray', alpha=0.3, edgecolor='black', linewidth=2)
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Structure Visualization')
    
    if show_nodes:
        ax.legend()
    
    plt.tight_layout()
    return fig, ax


def generate_voronoi_points(region_corners: np.ndarray, n_points: int,
                           seed: Optional[int] = None) -> np.ndarray:
    """
    Generate random points inside polygon for Voronoi tessellation.
    
    Parameters:
        region_corners: Nx2 array of polygon vertices
        n_points: Number of points to generate
        seed: Random seed for reproducibility
        
    Returns:
        Mx2 array of points inside polygon
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Bounding box
    xmin, ymin = region_corners.min(axis=0)
    xmax, ymax = region_corners.max(axis=0)
    
    # Generate points
    points = []
    while len(points) < n_points:
        p = np.array([
            np.random.uniform(xmin, xmax),
            np.random.uniform(ymin, ymax)
        ])
        
        if point_in_polygon(p, region_corners):
            points.append(p)
    
    return np.array(points)


def compute_element_stiffness_triangle(vertices: np.ndarray, 
                                      E: float, nu: float,
                                      thickness: float) -> np.ndarray:
    """
    Compute stiffness matrix for constant strain triangle (CST).
    
    Parameters:
        vertices: 3x2 array of vertex coordinates
        E: Young's modulus
        nu: Poisson's ratio
        thickness: Element thickness
        
    Returns:
        6x6 element stiffness matrix
    
    TODO: Implement actual CST formulation
    """
    # Placeholder
    K_elem = np.zeros((6, 6))
    
    # TODO: Implement CST stiffness matrix
    # 1. Compute B matrix (strain-displacement)
    # 2. Compute D matrix (constitutive)
    # 3. K = t * A * B^T * D * B
    
    return K_elem


def export_to_vtk(structure, filename: str):
    """
    Export structure to VTK format for ParaView visualization.
    
    Parameters:
        structure: Structure instance
        filename: Output filename (.vtu or .vtk)
    
    TODO: Implement VTK export
    """
    print(f"Exporting to {filename}...")
    print("WARNING: VTK export not yet implemented")
    
    # TODO: Use pyvista or meshio to export
    pass


def export_to_gmsh(structure, filename: str):
    """
    Export structure to Gmsh format.
    
    Parameters:
        structure: Structure instance
        filename: Output filename (.msh)
    
    TODO: Implement Gmsh export
    """
    print(f"Exporting to {filename}...")
    print("WARNING: Gmsh export not yet implemented")
    
    # TODO: Use meshio or direct gmsh API
    pass
