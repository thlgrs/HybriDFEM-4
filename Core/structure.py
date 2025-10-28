"""
Core abstractions for HybriDFEM structures.

This module defines the abstract base class for all structure types.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
import numpy as np


class Structure(ABC):
    """
    Abstract base class for all structure types.
    
    Defines the interface that all structures (block, FEM, hybrid) must implement.
    Handles DOF management, boundary conditions, and global assembly.
    """
    
    def __init__(self, name: str = "Structure"):
        """
        Initialize structure.
        
        Args:
            name: Structure identifier
        """
        self.name = name
        self.nb_dofs = 0
        self.dof_manager = None  # Will be DOFManager instance
        
        # Global matrices
        self.K = None  # Stiffness matrix
        self.M = None  # Mass matrix
        self.C = None  # Damping matrix
        
        # Boundary conditions and loads
        self.fixed_dofs = []
        self.prescribed_displacements = {}
        self.external_forces = None
        
        # Analysis state
        self.U = None  # Current displacements
        self.P_r = None  # Current resisting forces
        
    # =========================================================================
    # Abstract methods - must be implemented by subclasses
    # =========================================================================
    
    @abstractmethod
    def assemble_stiffness(self) -> np.ndarray:
        """
        Assemble global stiffness matrix.
        
        Returns:
            K: Global stiffness matrix (nb_dofs × nb_dofs)
        """
        pass
    
    @abstractmethod
    def assemble_mass(self) -> np.ndarray:
        """
        Assemble global mass matrix.
        
        Returns:
            M: Global mass matrix (nb_dofs × nb_dofs)
        """
        pass
    
    @abstractmethod
    def compute_internal_forces(self, U: np.ndarray) -> np.ndarray:
        """
        Compute internal resisting forces for given displacements.
        
        Args:
            U: Global displacement vector
            
        Returns:
            P_r: Internal resisting forces
        """
        pass
    
    @abstractmethod
    def get_element_stresses(self, U: np.ndarray) -> Dict:
        """
        Compute stresses in all elements.
        
        Args:
            U: Global displacement vector
            
        Returns:
            stresses: Dictionary mapping element IDs to stress tensors
        """
        pass
    
    # =========================================================================
    # Concrete methods - common to all structures
    # =========================================================================
    
    def apply_boundary_conditions(self, fixed_dofs: List[int], 
                                  prescribed: Optional[Dict[int, float]] = None):
        """
        Apply boundary conditions to structure.
        
        Args:
            fixed_dofs: List of DOF indices to fix (zero displacement)
            prescribed: Dictionary {dof_index: prescribed_value}
        """
        self.fixed_dofs = fixed_dofs
        if prescribed:
            self.prescribed_displacements = prescribed
    
    def apply_external_forces(self, forces: np.ndarray):
        """
        Apply external force vector.
        
        Args:
            forces: External force vector (nb_dofs,)
        """
        if forces.shape[0] != self.nb_dofs:
            raise ValueError(f"Force vector size {forces.shape[0]} doesn't match DOFs {self.nb_dofs}")
        self.external_forces = forces.copy()
    
    def enforce_boundary_conditions(self, K: np.ndarray, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enforce boundary conditions on system of equations.
        
        Uses penalty method for fixed DOFs.
        
        Args:
            K: Stiffness matrix
            F: Force vector
            
        Returns:
            K_modified, F_modified: Modified system
        """
        K_mod = K.copy()
        F_mod = F.copy()
        
        # Apply fixed DOFs (penalty method)
        penalty = 1e15 * np.max(np.abs(K.diagonal()))
        for dof in self.fixed_dofs:
            K_mod[dof, dof] += penalty
            F_mod[dof] = 0.0
        
        # Apply prescribed displacements
        for dof, value in self.prescribed_displacements.items():
            K_mod[dof, dof] += penalty
            F_mod[dof] = penalty * value
        
        return K_mod, F_mod
    
    def get_free_dofs(self) -> np.ndarray:
        """
        Get list of free (unconstrained) DOF indices.
        
        Returns:
            free_dofs: Array of free DOF indices
        """
        all_dofs = np.arange(self.nb_dofs)
        constrained = list(self.fixed_dofs) + list(self.prescribed_displacements.keys())
        free_dofs = np.setdiff1d(all_dofs, constrained)
        return free_dofs
    
    def get_reaction_forces(self, U: np.ndarray) -> np.ndarray:
        """
        Compute reaction forces at constrained DOFs.
        
        Args:
            U: Global displacement vector
            
        Returns:
            reactions: Reaction force vector
        """
        K = self.assemble_stiffness()
        P_int = self.compute_internal_forces(U)
        reactions = K @ U - P_int
        
        # Zero out free DOFs
        reactions_constrained = np.zeros_like(reactions)
        constrained_dofs = list(self.fixed_dofs) + list(self.prescribed_displacements.keys())
        reactions_constrained[constrained_dofs] = reactions[constrained_dofs]
        
        return reactions_constrained
    
    def update_state(self, U: np.ndarray):
        """
        Update structure state with new displacements.
        
        Args:
            U: New displacement vector
        """
        self.U = U.copy()
        self.P_r = self.compute_internal_forces(U)
    
    # =========================================================================
    # Information methods
    # =========================================================================
    
    def info(self) -> str:
        """
        Get structure information string.
        
        Returns:
            info_str: Formatted information about the structure
        """
        info = f"Structure: {self.name}\n"
        info += f"  Number of DOFs: {self.nb_dofs}\n"
        info += f"  Fixed DOFs: {len(self.fixed_dofs)}\n"
        info += f"  Prescribed DOFs: {len(self.prescribed_displacements)}\n"
        return info
    
    def validate(self) -> bool:
        """
        Validate structure configuration.
        
        Returns:
            valid: True if structure is properly configured
        """
        if self.nb_dofs == 0:
            print("Error: Structure has 0 DOFs")
            return False
        
        if self.external_forces is None:
            print("Warning: No external forces applied")
        
        # Check for sufficient constraints
        free_dofs = self.get_free_dofs()
        if len(free_dofs) == self.nb_dofs:
            print("Warning: No boundary conditions applied - structure may be unconstrained")
        
        return True


