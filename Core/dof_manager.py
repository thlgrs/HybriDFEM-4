from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
import numpy as np

class DOFManager:
    """
    Manages degree of freedom numbering and connectivity.

    Keeps track of:
    - Node to DOF mapping
    - Element to DOF mapping
    - Boundary condition DOFs
    """

    def __init__(self):
        self.node_to_dofs = {}  # {node_id: [dof1, dof2, ...]}
        self.dof_to_node = {}  # {dof_id: (node_id, local_dof)}
        self.total_dofs = 0

    def add_node(self, node_id: int, n_dofs: int) -> List[int]:
        """
        Add a node with specified number of DOFs.

        Args:
            node_id: Node identifier
            n_dofs: Number of DOFs at this node

        Returns:
            dof_indices: List of global DOF indices for this node
        """
        if node_id in self.node_to_dofs:
            raise ValueError(f"Node {node_id} already exists")

        dof_indices = list(range(self.total_dofs, self.total_dofs + n_dofs))
        self.node_to_dofs[node_id] = dof_indices

        for local_dof, global_dof in enumerate(dof_indices):
            self.dof_to_node[global_dof] = (node_id, local_dof)

        self.total_dofs += n_dofs
        return dof_indices

    def get_node_dofs(self, node_id: int) -> List[int]:
        """Get global DOF indices for a node."""
        return self.node_to_dofs.get(node_id, [])

    def get_element_dofs(self, node_ids: List[int]) -> List[int]:
        """Get all DOF indices for an element given its node IDs."""
        dofs = []
        for node_id in node_ids:
            dofs.extend(self.get_node_dofs(node_id))
        return dofs