"""
Relative Mooney Guidance Pipeline (Relative_Mooney_Guidance_Pipeline)
Transforms batch-level optimization from noisy absolute regression to robust relative guidance:
1. Raw Material & Baseline Mooney De-biasing
2. Golden Process Corridor Projection (Method A)
3. Self-Supervised State Embedding & Pairwise Delta (Method B)
4. Causal Sensitivity Inverse Constrained Optimization (Method C)
"""
__version__ = "1.0.0"
