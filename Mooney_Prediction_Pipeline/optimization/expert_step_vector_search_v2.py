"""Canonical corrected V2 entry point for expert Step-vector shadow search."""

from __future__ import annotations

from Mooney_Prediction_Pipeline.optimization import expert_step_vector_search as _core
from Mooney_Prediction_Pipeline.optimization.expert_shadow_policy_v2 import policy_reasons

# The core search resolves this global at call time.  V2 injects the corrected
# indexed-vector gate without changing the retained historical implementation.
_core.policy_reasons = policy_reasons

ExpertVectorSearchResult = _core.ExpertVectorSearchResult
search_expert_step_vectors = _core.search_expert_step_vectors
