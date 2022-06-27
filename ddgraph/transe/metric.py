import random
from typing import List, NamedTuple, Tuple

import torch

import ddgraph.graph as graph
import ddgraph.transe.transe as transe


class MetricsBundle(NamedTuple):
    mean_rank: float
    hits_at_10: float


class Calculator:
    _HEADS = 0
    _TAILS = 2
    
    _dataset: graph.TripletDataset
    _sample_size: int
    
    def __init__(self, dataset: graph.TripletDataset, sample_size: int = 256) -> None:
        self._dataset = dataset
        self._sample_size = sample_size

    def calculate(self, model: transe.TranseModel) -> MetricsBundle:
        cum_hits_at_10 = 0.0
        cum_rank = 0.0
        
        for _ in range(self._sample_size):
            triplet = self._dataset[random.randint(0, len(self._dataset) - 1)]

            # Corrupting heads.
            dists, original_idx = self._corrupted_dists(triplet, model, triplet_idx=self._HEADS)
            hits_at_10, rank = self._metrics_for_side(triplet, dists, original_idx, triplet_idx=self._HEADS)
            
            cum_hits_at_10 += hits_at_10
            cum_rank += rank
            
            # Corrupting tails.
            dists, original_idx = self._corrupted_dists(triplet, model, triplet_idx=self._TAILS)
            hits_at_10, rank = self._metrics_for_side(triplet, dists, original_idx, triplet_idx=self._TAILS)
            
            cum_hits_at_10 += hits_at_10
            cum_rank += rank

        hits_at_10 = cum_hits_at_10 / float(self._sample_size * 2)
        mean_rank = cum_rank / float(self._sample_size * 2)

        return MetricsBundle(mean_rank=mean_rank, hits_at_10=hits_at_10)

    def _corrupted_dists(self, triplet: torch.IntTensor, model: transe.TranseModel, triplet_idx: int) -> Tuple[List[float], int]:
        corrupted_triplet = torch.clone(triplet)

        # Corrupting heads.
        dists = []

        for j in range(self._dataset.entities_len()):
            if triplet[triplet_idx] == j:
                original_idx = j

            corrupted_triplet[triplet_idx] = j

            dist = model(torch.unsqueeze(corrupted_triplet, dim=0)).item()
            dists.append(dist)
        
        return dists, original_idx

    def _metrics_for_side(self, triplet: torch.IntTensor, dists: List[float], original_idx: int, triplet_idx: int) -> Tuple[float, float]:
        corrupted_triplet = torch.clone(triplet)

        closest_triplet_indices = self._closest_triplets_indices(dists, n=10)
        existing_count = 0

        for j in closest_triplet_indices:
            corrupted_triplet[triplet_idx] = j
            
            if self._dataset.exists(corrupted_triplet):
                existing_count += 1

        hits_at_10 = float(existing_count) / 10.0
        rank = float(self._triplet_rank(dists, original_idx))

        return hits_at_10, rank

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
