import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings

# Suppress sklearn warnings if needed
warnings.filterwarnings("ignore")

try:
    from transformer_lens import HookedTransformer
except ImportError:
    HookedTransformer = None
    print("Warning: transformer_lens not found. Cortex execution will fail.")

class AttractorAnalyzer:
    def __init__(self, model_name: str = "gpt2-small"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.model_name = model_name
        self.pca = PCA(n_components=3)
        self.is_fitted = False
        
        # Anchors for classification
        self.safe_centroid: Optional[np.ndarray] = None
        self.safe_covariance: Optional[np.ndarray] = None
        self.safe_inv_cov: Optional[np.ndarray] = None
        self.unsafe_centroid: Optional[np.ndarray] = None
        self.calibration_data: List[np.ndarray] = []

    def load_model(self):
        """Loads the HookedTransformer model."""
        if self.model is None and HookedTransformer:
            print(f"Loading {self.model_name} on {self.device}...")
            self.model = HookedTransformer.from_pretrained(
                self.model_name, 
                device=self.device
            )

    def get_thought_trajectory(self, prompt: str) -> np.ndarray:
        """
        Runs the model and extracts the residual stream activation 
        at the last token for every layer.
        output shape: (n_layers, n_embd)
        """
        if self.model is None:
            self.load_model()
        
        # We handle the case where model loading might have failed
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        _, cache = self.model.run_with_cache(prompt)
        
        trajectory = []
        # Layer 0 is the embedding layer, usually hooked as 'blocks.0.hook_resid_pre' 
        # or we just iterate through blocks.
        # transformer_lens standard: blocks.{layer}.hook_resid_post
        
        n_layers = self.model.cfg.n_layers
        
        for layer in range(n_layers):
            # cache key
            key = f"blocks.{layer}.hook_resid_post"
            # Shape: [batch, pos, dim]. We want [0, -1, :]
            # detaching from graph to save memory
            state = cache[key][0, -1, :].detach().cpu().numpy()
            trajectory.append(state)
            
        return np.array(trajectory)

    def generate_response(self, prompt: str, max_tokens: int = 30) -> str:
        """
        Generates a text completion for the given prompt.
        """
        if self.model is None:
            self.load_model()
            
        try:
            # Simple generation
            output = self.model.generate(prompt, max_new_tokens=max_tokens, temperature=0.7, verbose=False)
            # Return only the new text if possible, or full text
            return output
        except Exception as e:
            return f"[Generation Error: {e}]"

    def fit_pca(self, trajectories: List[np.ndarray]):
        """
        Fits privacy-preserving PCA on a batch of trajectories.
        Stack all layers from all trajectories to define the global 3D space.
        """
        # data shape: (n_samples * n_layers, n_embd)
        combined_data = np.vstack(trajectories)
        self.pca.fit(combined_data)
        self.is_fitted = True

    def transform_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Projects a High-D trajectory (n_layers, n_embd) into 3D (n_layers, 3).
        """
        if not self.is_fitted:
            raise RuntimeError("PCA is not fitted. Run calibration first.")
        
        return self.pca.transform(trajectory)

    def calibrate(self, safe_prompts: List[str], unsafe_prompts: List[str]):
        """
        Runs a set of prompts to fit PCA and define centroids.
        """
        print("Calibrating Cortex Monitor...")
        safe_trajs = [self.get_thought_trajectory(p) for p in safe_prompts]
        unsafe_trajs = [self.get_thought_trajectory(p) for p in unsafe_prompts]
        
        all_trajs = safe_trajs + unsafe_trajs
        self.fit_pca(all_trajs)
        
        # Calculate centroids in 3D space based on the FINAL layer
        # as the attractor state determines the output.
        safe_3d = [self.transform_trajectory(t)[-1] for t in safe_trajs]
        unsafe_3d = [self.transform_trajectory(t)[-1] for t in unsafe_trajs]
        
        self.safe_centroid = np.mean(safe_3d, axis=0)
        self.unsafe_centroid = np.mean(unsafe_3d, axis=0)
        
        # Calculate Covariance for Mahalanobis Distance (on 3D projected data for stability)
        # Using 3D because n_samples (5-10) < n_features (768) would yield singular matrix on raw data
        safe_data = np.array(safe_3d)
        # Regularization to prevent singular matrix
        reg = 1e-6 * np.eye(safe_data.shape[1])
        self.safe_covariance = np.cov(safe_data, rowvar=False) + reg
        self.safe_inv_cov = np.linalg.inv(self.safe_covariance)
        
        print(f"Calibration Complete. Anchors set. Safe Covariance shape: {self.safe_covariance.shape}")

    def compute_mahalanobis(self, vector_3d: np.ndarray) -> float:
        """
        Calculates Mahalanobis distance from the Safe distribution.
        D_M(x) = sqrt( (x-u)T * Cov^-1 * (x-u) )
        """
        if self.safe_centroid is None or self.safe_inv_cov is None:
            return 0.0
            
        delta = vector_3d - self.safe_centroid
        # Distance squared
        d_squared = np.dot(np.dot(delta, self.safe_inv_cov), delta.T)
        return float(np.sqrt(d_squared))

    def classify_state(self, trajectory_3d: np.ndarray) -> str:
        """
        Classifies a geometric trajectory into:
        - Safe: Converges to Safe Basin
        - Unsafe: Converges to Dark Attractor
        - Confused: Limit Cycle / High Path Length (Chaotic)
        """
        if self.safe_centroid is None or self.unsafe_centroid is None:
            return "Uncalibrated"

        final_state = trajectory_3d[-1]
        
        # 1. Distances to attractors
        d_safe = np.linalg.norm(final_state - self.safe_centroid)
        d_unsafe = np.linalg.norm(final_state - self.unsafe_centroid)
        
        # 2. Check for Chaos/Confusion (Limit Cycle)
        # We measure the "path length" in the final few layers. 
        # If it's spinning but not converging, path length is high relative to displacement.
        # Simple heuristic: total path variance or erratic movement.
        # For this MVP: if it's far from BOTH, it's confused/unknown.
        
        threshold = 10.0 # Arbitrary distance threshold, would tune in derived class
        
        if d_unsafe < d_safe and d_unsafe < threshold:
            return "Unsafe"
        elif d_safe < d_unsafe and d_safe < threshold:
            return "Safe"
        else:
            return "Confused"

    def cluster_behaviors(self, history: List[Dict[str, Any]], n_clusters: int = 3) -> Dict[int, List[str]]:
        """
        Groups prompts based on the similarity of their thought trajectories.
        """
        if len(history) < n_clusters:
            return {0: [h['prompt'] for h in history]}
            
        # Extract features: Flattened 3D trajectory
        # Shape: [n_samples, n_layers * 3]
        features = []
        prompts = []
        
        for item in history:
            # item['coords'] is (n_layers, 3)
            traj_flat = item['coords'].flatten()
            features.append(traj_flat)
            prompts.append(item['prompt'])
            
        features = np.array(features)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(features)
        
        clusters = {}
        for label, prompt in zip(labels, prompts):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(prompt)
            
        return clusters
