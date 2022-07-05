import random
from typing import List, NamedTuple, Tuple

import torch

import ddgraph.graph as graph
import ddgraph.transh.transh as transh


class MetricsBundle(NamedTuple):
    mean_rank: float
    hits_at_10: float


class Calculator:
    _HEADS = 0
    _TAILS = 2
    
    _onto: graph.Ontology
    _sample_size: int
    
    def __init__(self, onto: graph.Ontology, sample_size: int = 256) -> None:
        self._onto = onto
        self._sample_size = sample_size

    @torch.no_grad()
    def calculate(self, model: transh.TranshModel) -> MetricsBundle:
        cum_hits_at_10 = 0.0
        cum_rank = 0.0

        for _ in range(self._sample_size):
            raw_triplet = self._onto.get_triplet(random.randint(0, self._onto.triplets_len() - 1))
            triplet = graph.tensorify_triplet(raw_triplet)

            # Corrupting heads.
            dists, original_idx = self._corrupted_dists(triplet, model, part_idx=self._HEADS)

            if self._is_hit_at_10(dists, original_idx):
                cum_hits_at_10 += 1

            cum_rank += float(self._triplet_rank(dists, original_idx))

            # Corrupting tails.
            dists, original_idx = self._corrupted_dists(triplet, model, part_idx=self._TAILS)
            
            if self._is_hit_at_10(dists, original_idx):
                cum_hits_at_10 += 1
            
            cum_rank += float(self._triplet_rank(dists, original_idx))

        hits_at_10 = cum_hits_at_10 / float(self._sample_size * 2)
        mean_rank = cum_rank / float(self._sample_size * 2)

        return MetricsBundle(mean_rank=mean_rank, hits_at_10=hits_at_10)

    def _corrupted_dists(self, triplet: torch.IntTensor, model: transh.TranshModel, part_idx: int) -> Tuple[List[float], int]:
        corrupted_triplet = torch.clone(triplet)

        # Corrupting heads.
        dists = []

        for j in range(self._onto.entities_len()):
            if triplet[part_idx] == j:
                original_idx = j

            corrupted_triplet[part_idx] = j

            dist = model(torch.unsqueeze(corrupted_triplet, dim=0)).item()
            dists.append(dist)
        
        return dists, original_idx

    def _is_hit_at_10(self, dists: List[float], original_idx: int) -> Tuple[float, float]:
        return original_idx in self._closest_triplets_indices(dists, n=10)

    def _closest_triplets_indices(self, dists: List[float], n: int) -> List[int]:
        closest_triplets_indices = []

        # Using insertion sort logic as we only need the closest n items
        # which are usually 5, 10 or 25.
        for i in range(n):
            min_dist = dists[i]
            min_idx = i

            for j, dist in enumerate(dists[i:]):
                if min_dist > dist:
                    min_dist = dist
                    min_idx = j

            closest_triplets_indices.append(min_idx)
            dists[i], dists[min_idx] = dists[min_idx], dists[i]

        return closest_triplets_indices

    def _triplet_rank(self, dists: List[float], idx: int) -> int:
        closer_count = 0
        
        for i, dist in enumerate(dists):
            if i == idx:
                continue

            if dist < dists[idx]:
                closer_count += 1

        return closer_count
