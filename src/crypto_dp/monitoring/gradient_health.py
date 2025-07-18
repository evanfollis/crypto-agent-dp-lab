"""
Advanced gradient health monitoring infrastructure for E2E-DP.

Based on CLAUDE.md specifications for comprehensive gradient diagnostics
including variance of absolute gradients, signal-to-total-variance ratio,
and layer-wise analysis.
"""

import jax
import jax.numpy as jnp
from jax import tree_util
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass, field
import time


@dataclass
class GradientMetrics:
    """Container for gradient health metrics."""
    norm_ratio: float
    signal_to_total_variance: float
    variance_abs_gradients: float
    gradient_sparsity: float
    variance_trend: Optional[float] = None
    layer_norms: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def is_healthy(self) -> Tuple[bool, List[str]]:
        """Check if metrics indicate healthy gradients."""
        issues = []
        
        # Check norm ratio (should be in [0.1, 10])
        if self.norm_ratio < 0.1:
            issues.append(f"Vanishing gradient flow (norm ratio: {self.norm_ratio:.3f})")
        elif self.norm_ratio > 10:
            issues.append(f"Exploding gradient flow (norm ratio: {self.norm_ratio:.3f})")
        
        # Check signal-to-total-variance
        if self.signal_to_total_variance < 0.01:
            issues.append(f"Poor gradient signal (STV: {self.signal_to_total_variance:.3f})")
        
        # Check sparsity
        if self.gradient_sparsity > 0.9:
            issues.append(f"Excessive gradient sparsity ({self.gradient_sparsity:.1%})")
        
        # Check variance trend if available
        if self.variance_trend is not None and abs(self.variance_trend) > 0.1:
            direction = "increasing" if self.variance_trend > 0 else "decreasing"
            issues.append(f"Gradient variance {direction} (trend: {self.variance_trend:.3f})")
        
        return len(issues) == 0, issues


class EnhancedGradientMonitor:
    """
    Enhanced gradient health monitoring with advanced metrics from CLAUDE.md.
    """
    
    def __init__(self, window_size: int = 100, track_layers: bool = True):
        """
        Initialize gradient monitor.
        
        Args:
            window_size: Number of recent measurements to track
            track_layers: Whether to track layer-wise metrics
        """
        self.window_size = window_size
        self.track_layers = track_layers
        self.history: List[GradientMetrics] = []
        self.variance_history: List[float] = []
        
    def compute_metrics(self, grads: Any, prefix: str = "") -> GradientMetrics:
        """
        Compute comprehensive gradient metrics.
        
        Args:
            grads: PyTree of gradients
            prefix: Optional prefix for nested structures
            
        Returns:
            GradientMetrics object with all computed metrics
        """
        # Flatten gradients and get structure info
        flat_grads, tree_def = tree_util.tree_flatten_with_path(grads)
        
        # Separate by layers if tracking enabled
        layer_grads = {}
        layer_norms = {}
        
        for path, grad in flat_grads:
            if grad is None:
                continue
                
            # Extract layer name from path
            layer_name = self._path_to_layer_name(path, prefix)
            
            if self.track_layers:
                if layer_name not in layer_grads:
                    layer_grads[layer_name] = []
                layer_grads[layer_name].append(grad.flatten())
                
        # Compute layer norms
        for layer_name, grads_list in layer_grads.items():
            layer_grad = jnp.concatenate(grads_list)
            layer_norms[layer_name] = float(jnp.linalg.norm(layer_grad))
        
        # Get all gradients as single array
        all_grads = jnp.concatenate([g.flatten() for _, g in flat_grads if g is not None])
        
        # Compute norm ratio between first and last layers
        layer_names = list(layer_norms.keys())
        if len(layer_names) >= 2:
            first_norm = layer_norms[layer_names[0]]
            last_norm = layer_norms[layer_names[-1]]
            norm_ratio = last_norm / (first_norm + 1e-8)
        else:
            norm_ratio = 1.0
        
        # Signal-to-total-variance ratio (prevents cancellation)
        batch_dim = 0  # Assuming first dimension is batch
        if all_grads.ndim > 1:
            mean_over_batch = jnp.mean(all_grads, axis=batch_dim)
            var_of_mean = jnp.var(mean_over_batch)
            mean_of_var = jnp.mean(jnp.var(all_grads, axis=batch_dim))
            signal_to_total_variance = var_of_mean / (mean_of_var + 1e-8)
        else:
            signal_to_total_variance = jnp.var(all_grads) / (jnp.var(all_grads) + 1e-8)
        
        # Variance of absolute gradients (prevents positive/negative cancellation)
        variance_abs_gradients = float(jnp.var(jnp.abs(all_grads)))
        
        # Gradient sparsity
        gradient_sparsity = float(jnp.mean(jnp.abs(all_grads) < 1e-6))
        
        # Store variance for trend analysis
        current_variance = float(jnp.var(all_grads))
        self.variance_history.append(current_variance)
        if len(self.variance_history) > self.window_size:
            self.variance_history.pop(0)
        
        # Compute variance trend if enough history
        variance_trend = None
        if len(self.variance_history) >= 10:
            # Simple linear regression on log variance
            x = np.arange(len(self.variance_history))
            y = np.log(np.array(self.variance_history) + 1e-8)
            variance_trend = float(np.polyfit(x, y, 1)[0])
        
        metrics = GradientMetrics(
            norm_ratio=float(norm_ratio),
            signal_to_total_variance=float(signal_to_total_variance),
            variance_abs_gradients=variance_abs_gradients,
            gradient_sparsity=gradient_sparsity,
            variance_trend=variance_trend,
            layer_norms=layer_norms
        )
        
        # Add to history
        self.history.append(metrics)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        
        return metrics
    
    def _path_to_layer_name(self, path: Tuple[Any, ...], prefix: str) -> str:
        """Convert JAX path to readable layer name."""
        parts = []
        if prefix:
            parts.append(prefix)
            
        for key in path:
            # Handle both old and new JAX path formats
            if hasattr(key, 'key'):
                parts.append(str(key.key))
            else:
                parts.append(str(key))
                
        return ".".join(parts) if parts else "root"
    
    def get_summary_stats(self) -> Dict[str, float]:
        """Get summary statistics over recent history."""
        if not self.history:
            return {}
        
        recent = self.history[-20:]  # Last 20 measurements
        
        summary = {
            'mean_norm_ratio': np.mean([m.norm_ratio for m in recent]),
            'std_norm_ratio': np.std([m.norm_ratio for m in recent]),
            'mean_stv': np.mean([m.signal_to_total_variance for m in recent]),
            'mean_sparsity': np.mean([m.gradient_sparsity for m in recent]),
            'health_rate': np.mean([m.is_healthy()[0] for m in recent])
        }
        
        return summary
    
    def plot_history(self, save_path: Optional[str] = None):
        """Plot gradient health metrics over time."""
        if not self.history:
            return
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Gradient Health Monitoring')
        
        # Norm ratios
        norm_ratios = [m.norm_ratio for m in self.history]
        axes[0, 0].plot(norm_ratios)
        axes[0, 0].axhline(y=0.1, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].axhline(y=10, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].set_ylabel('Norm Ratio')
        axes[0, 0].set_yscale('log')
        axes[0, 0].set_title('Gradient Flow (First/Last Layer)')
        
        # Signal-to-total-variance
        stvs = [m.signal_to_total_variance for m in self.history]
        axes[0, 1].plot(stvs)
        axes[0, 1].axhline(y=0.01, color='r', linestyle='--', alpha=0.5)
        axes[0, 1].set_ylabel('STV Ratio')
        axes[0, 1].set_yscale('log')
        axes[0, 1].set_title('Signal-to-Total-Variance')
        
        # Variance of absolute gradients
        var_abs = [m.variance_abs_gradients for m in self.history]
        axes[1, 0].plot(var_abs)
        axes[1, 0].set_ylabel('Variance')
        axes[1, 0].set_yscale('log')
        axes[1, 0].set_title('Variance of |Gradients|')
        
        # Sparsity
        sparsity = [m.gradient_sparsity for m in self.history]
        axes[1, 1].plot(sparsity)
        axes[1, 1].axhline(y=0.9, color='r', linestyle='--', alpha=0.5)
        axes[1, 1].set_ylabel('Sparsity')
        axes[1, 1].set_title('Gradient Sparsity')
        
        for ax in axes.flat:
            ax.set_xlabel('Training Step')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        else:
            plt.show()


class GradientClipTracker:
    """Track gradient clipping statistics."""
    
    def __init__(self):
        self.clip_history = []
        self.norm_history = []
    
    def track_clip(self, global_norm: float, max_norm: float, clipped: bool):
        """Track a gradient clipping event."""
        self.norm_history.append(float(global_norm))
        self.clip_history.append(clipped)
        
        # Keep only recent history
        if len(self.norm_history) > 1000:
            self.norm_history.pop(0)
            self.clip_history.pop(0)
    
    def get_clip_rate(self, window: int = 100) -> float:
        """Get recent clipping rate."""
        if len(self.clip_history) < window:
            recent = self.clip_history
        else:
            recent = self.clip_history[-window:]
        
        return np.mean(recent) if recent else 0.0
    
    def get_norm_stats(self) -> Dict[str, float]:
        """Get gradient norm statistics."""
        if not self.norm_history:
            return {}
        
        return {
            'mean_norm': np.mean(self.norm_history),
            'std_norm': np.std(self.norm_history),
            'max_norm': np.max(self.norm_history),
            'p95_norm': np.percentile(self.norm_history, 95)
        }


def apply_global_gradient_clip(
    grads: Any,
    max_norm: float = 10.0,
    clip_tracker: Optional[GradientClipTracker] = None
) -> Tuple[Any, bool]:
    """
    Apply global gradient norm clipping with tracking.
    
    Args:
        grads: PyTree of gradients
        max_norm: Maximum allowed global norm
        clip_tracker: Optional tracker for clipping statistics
        
    Returns:
        Clipped gradients and whether clipping occurred
    """
    # Compute global norm
    leaves = tree_util.tree_leaves(grads)
    global_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in leaves if g is not None))
    
    # Compute clip factor
    clip_factor = jnp.minimum(1.0, max_norm / (global_norm + 1e-8))
    clip_occurred = global_norm > max_norm
    
    # Track if requested
    if clip_tracker is not None:
        clip_tracker.track_clip(float(global_norm), max_norm, bool(clip_occurred))
    
    # Apply clipping
    clipped_grads = tree_util.tree_map(lambda g: g * clip_factor if g is not None else None, grads)
    
    return clipped_grads, bool(clip_occurred)


if __name__ == "__main__":
    # Example usage with synthetic gradients
    print("Testing enhanced gradient monitoring...")
    
    # Create monitor
    monitor = EnhancedGradientMonitor()
    
    # Simulate training with varying gradient health
    key = jax.random.PRNGKey(42)
    
    for step in range(200):
        key, subkey = jax.random.split(key)
        
        # Simulate gradients with varying properties
        if step < 50:
            # Healthy gradients
            scale = 1.0
        elif step < 100:
            # Vanishing gradients
            scale = 0.001 ** (step / 100)
        elif step < 150:
            # Exploding gradients
            scale = 10 ** ((step - 100) / 50)
        else:
            # Recovery
            scale = 1.0
        
        # Create synthetic gradient structure
        grads = {
            'layer1': {
                'weight': scale * jax.random.normal(subkey, (32, 16)),
                'bias': scale * jax.random.normal(jax.random.split(subkey)[0], (32,))
            },
            'layer2': {
                'weight': scale * 0.5 * jax.random.normal(jax.random.split(subkey)[1], (16, 8)),
                'bias': scale * 0.5 * jax.random.normal(jax.random.split(subkey, 3)[2], (16,))
            }
        }
        
        # Compute metrics
        metrics = monitor.compute_metrics(grads)
        
        if step % 50 == 0:
            is_healthy, issues = metrics.is_healthy()
            print(f"\nStep {step}:")
            print(f"  Healthy: {is_healthy}")
            if not is_healthy:
                print(f"  Issues: {', '.join(issues)}")
            print(f"  Norm ratio: {metrics.norm_ratio:.3f}")
            print(f"  STV: {metrics.signal_to_total_variance:.3e}")
            print(f"  Sparsity: {metrics.gradient_sparsity:.1%}")
    
    # Summary statistics
    print("\nSummary statistics:")
    for key, value in monitor.get_summary_stats().items():
        print(f"  {key}: {value:.3f}")
    
    # Save gradient health plot
    monitor.plot_history(save_path="gradient_health.png")
    print("\nGradient health plot saved to gradient_health.png")