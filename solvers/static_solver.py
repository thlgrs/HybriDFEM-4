"""
Static solvers for nonlinear structural analysis.

Implements:
- Newton-Raphson
- Modified Newton-Raphson  
- Arc-length (Riks method)
- Displacement control
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Callable
import numpy as np
from scipy.sparse import linalg as sp_linalg


class StaticSolver(ABC):
    """
    Abstract base class for static nonlinear solvers.
    """
    
    def __init__(self, max_iter: int = 50, tolerance: float = 1e-6):
        """
        Initialize solver.
        
        Args:
            max_iter: Maximum iterations per load step
            tolerance: Convergence tolerance on residual norm
        """
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.observers = []  # Observer pattern for monitoring
        
        # Solution history
        self.load_factors = []
        self.displacements = []
        self.iterations_per_step = []
    
    def attach_observer(self, observer):
        """Attach an observer for monitoring progress."""
        self.observers.append(observer)
    
    def _notify_step_start(self, step: int, lambda_val: float):
        """Notify observers of step start."""
        for obs in self.observers:
            if hasattr(obs, 'on_step_start'):
                obs.on_step_start(step, lambda_val)
    
    def _notify_iteration(self, iter_num: int, residual_norm: float):
        """Notify observers of iteration."""
        for obs in self.observers:
            if hasattr(obs, 'on_iteration'):
                obs.on_iteration(iter_num, residual_norm)
    
    def _notify_convergence(self, U: np.ndarray, P_r: np.ndarray):
        """Notify observers of convergence."""
        for obs in self.observers:
            if hasattr(obs, 'on_convergence'):
                obs.on_convergence(U.copy(), P_r.copy())
    
    @abstractmethod
    def solve(self, structure, P_ref: np.ndarray, n_steps: int = 10) -> dict:
        """
        Solve the nonlinear problem.
        
        Args:
            structure: Structure object
            P_ref: Reference load vector
            n_steps: Number of load steps
            
        Returns:
            results: Dictionary with solution data
        """
        pass


class NewtonRaphsonSolver(StaticSolver):
    """
    Newton-Raphson solver with force control.
    
    Solves: P_r(U_i) = λ_i * P_ref
    
    At each step:
    1. Increment load factor: λ_i = λ_(i-1) + Δλ
    2. Iterate: K_tan * ΔU = λ_i * P_ref - P_r(U)
    3. Update: U = U + ΔU
    4. Check convergence: ||R|| < tol
    """
    
    def __init__(self, max_iter: int = 50, tolerance: float = 1e-6,
                 modified: bool = False):
        """
        Initialize Newton-Raphson solver.
        
        Args:
            max_iter: Maximum iterations per step
            tolerance: Convergence tolerance
            modified: If True, use Modified NR (constant stiffness)
        """
        super().__init__(max_iter, tolerance)
        self.modified = modified
    
    def solve(self, structure, P_ref: np.ndarray, n_steps: int = 10) -> dict:
        """
        Solve using Newton-Raphson method.
        
        Args:
            structure: Structure object
            P_ref: Reference load vector
            n_steps: Number of load steps
            
        Returns:
            results: Solution history
        """
        # Initialize
        U = np.zeros(structure.nb_dofs)
        lambda_total = 0.0
        d_lambda = 1.0 / n_steps
        
        # Initial stiffness (for modified NR)
        if self.modified:
            K_initial = structure.assemble_stiffness()
        
        # Load stepping
        for step in range(n_steps):
            lambda_total += d_lambda
            self._notify_step_start(step, lambda_total)
            
            # Newton-Raphson iteration
            converged = False
            for iter_num in range(self.max_iter):
                # Compute internal forces
                P_r = structure.compute_internal_forces(U)
                
                # Residual
                R = lambda_total * P_ref - P_r
                
                # Apply boundary conditions to residual
                free_dofs = structure.get_free_dofs()
                R_free = R[free_dofs]
                
                # Check convergence
                residual_norm = np.linalg.norm(R_free)
                self._notify_iteration(iter_num, residual_norm)
                
                if residual_norm < self.tolerance:
                    converged = True
                    break
                
                # Compute tangent stiffness
                if self.modified:
                    K_tan = K_initial
                else:
                    K_tan = structure.assemble_stiffness()
                
                # Apply boundary conditions
                K_mod, R_mod = structure.enforce_boundary_conditions(K_tan, R)
                
                # Solve for displacement increment
                try:
                    delta_U = np.linalg.solve(K_mod, R_mod)
                except np.linalg.LinAlgError:
                    print(f"Singular stiffness matrix at step {step}, iteration {iter_num}")
                    raise
                
                # Update displacements
                U += delta_U
            
            if not converged:
                print(f"Warning: Step {step} did not converge after {self.max_iter} iterations")
            
            # Store results
            self.load_factors.append(lambda_total)
            self.displacements.append(U.copy())
            self.iterations_per_step.append(iter_num + 1)
            
            # Update structure state
            structure.update_state(U)
            
            # Notify observers
            P_r = structure.compute_internal_forces(U)
            self._notify_convergence(U, P_r)
        
        return {
            'load_factors': np.array(self.load_factors),
            'displacements': np.array(self.displacements),
            'iterations': self.iterations_per_step,
            'converged': converged
        }


class ArcLengthSolver(StaticSolver):
    """
    Arc-length solver (Riks method) for path-following.
    
    Suitable for:
    - Snap-through problems
    - Snap-back problems
    - Post-buckling analysis
    
    Constraint equation: ||ΔU||² + ψ² * Δλ² = Δs²
    """
    
    def __init__(self, max_iter: int = 50, tolerance: float = 1e-6,
                 arc_length: float = 1.0, psi: float = 1.0):
        """
        Initialize arc-length solver.
        
        Args:
            max_iter: Maximum iterations per step
            tolerance: Convergence tolerance
            arc_length: Arc-length parameter Δs
            psi: Scaling factor for load vs displacement
        """
        super().__init__(max_iter, tolerance)
        self.arc_length = arc_length
        self.psi = psi
    
    def solve(self, structure, P_ref: np.ndarray, n_steps: int = 10) -> dict:
        """
        Solve using arc-length method.
        
        Args:
            structure: Structure object
            P_ref: Reference load vector
            n_steps: Number of load steps
            
        Returns:
            results: Solution history
        """
        # Initialize
        U = np.zeros(structure.nb_dofs)
        lambda_total = 0.0
        
        # Load stepping with arc-length control
        for step in range(n_steps):
            self._notify_step_start(step, lambda_total)
            
            # Predictor step
            K = structure.assemble_stiffness()
            delta_U_bar = np.linalg.solve(K, P_ref)  # Displacement for unit load
            
            # Initial increments
            delta_lambda = self.arc_length / np.sqrt(
                np.dot(delta_U_bar, delta_U_bar) + self.psi**2
            )
            delta_U = delta_lambda * delta_U_bar
            
            U_trial = U + delta_U
            lambda_trial = lambda_total + delta_lambda
            
            # Corrector iterations
            converged = False
            for iter_num in range(self.max_iter):
                # Compute internal forces
                P_r = structure.compute_internal_forces(U_trial)
                
                # Residual
                R = lambda_trial * P_ref - P_r
                
                # Check convergence
                residual_norm = np.linalg.norm(R)
                self._notify_iteration(iter_num, residual_norm)
                
                if residual_norm < self.tolerance:
                    converged = True
                    break
                
                # Tangent stiffness
                K_tan = structure.assemble_stiffness()
                
                # Solve two systems
                delta_U_1 = np.linalg.solve(K_tan, R)
                delta_U_2 = np.linalg.solve(K_tan, P_ref)
                
                # Arc-length constraint to find delta_lambda
                a = np.dot(delta_U_2, delta_U_2) + self.psi**2
                b = 2 * (np.dot(delta_U, delta_U_2) + self.psi**2 * delta_lambda)
                c = np.dot(delta_U, delta_U) + self.psi**2 * delta_lambda**2 - self.arc_length**2
                
                # Solve quadratic
                discriminant = b**2 - 4*a*c
                if discriminant < 0:
                    print(f"Warning: Negative discriminant at step {step}")
                    break
                
                delta_lambda_corr = (-b + np.sqrt(discriminant)) / (2*a)
                
                # Update
                U_trial += delta_U_1 + delta_lambda_corr * delta_U_2
                lambda_trial += delta_lambda_corr
                
                delta_U = U_trial - U
                delta_lambda = lambda_trial - lambda_total
            
            if not converged:
                print(f"Warning: Step {step} did not converge")
            
            # Accept step
            U = U_trial
            lambda_total = lambda_trial
            
            # Store results
            self.load_factors.append(lambda_total)
            self.displacements.append(U.copy())
            self.iterations_per_step.append(iter_num + 1)
            
            structure.update_state(U)
            P_r = structure.compute_internal_forces(U)
            self._notify_convergence(U, P_r)
        
        return {
            'load_factors': np.array(self.load_factors),
            'displacements': np.array(self.displacements),
            'iterations': self.iterations_per_step,
            'converged': converged
        }


class DisplacementControlSolver(StaticSolver):
    """
    Displacement-controlled solver.
    
    Prescribes displacement at a control DOF and solves for reactions.
    Useful for:
    - Softening problems
    - Post-peak behavior
    """
    
    def __init__(self, control_dof: int, max_iter: int = 50, tolerance: float = 1e-6):
        """
        Initialize displacement control solver.
        
        Args:
            control_dof: DOF index to control
            max_iter: Maximum iterations per step
            tolerance: Convergence tolerance
        """
        super().__init__(max_iter, tolerance)
        self.control_dof = control_dof
    
    def solve(self, structure, u_max: float, n_steps: int = 10) -> dict:
        """
        Solve with displacement control.
        
        Args:
            structure: Structure object
            u_max: Maximum displacement at control DOF
            n_steps: Number of displacement steps
            
        Returns:
            results: Solution history
        """
        # Implementation similar to NR but with displacement prescription
        # at control_dof
        raise NotImplementedError("Displacement control to be implemented")
