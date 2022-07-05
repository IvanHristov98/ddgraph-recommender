import random
from typing import List, NamedTuple, Tuple

import torch

import ddgraph.graph as graph
import ddgraph.transh.transh as transh


class MetricsBundle(NamedTuple):
    mean_rank: float
    hits_at_5: float
    hits_at_10: float
    hits_at_20: float


class Calculator:
    _HEADS = 0
    _REL = 1
    _TAILS = 2
    
    _onto: graph.Ontology
    _sample_size: int
    
    def __init__(self, onto: graph.Ontology, sample_size: int = 256) -> None:
        self._onto = onto
        self._sample_size = sample_size

    @torch.no_grad()
    def calculate(self, model: transh.TranshModel) -> MetricsBundle:
        cum_hits_at_5 = 0.0
        cum_hits_at_10 = 0.0
        cum_hits_at_20 = 0.0
        cum_rank = 0.0

        for _ in range(self._sample_size):
            raw_triplet = self._onto.get_triplet(random.randint(0, self._onto.triplets_len() - 1))
            triplet = graph.tensorify_triplet(raw_triplet)

            # Corrupting tails.
            dists = self._corrupted_dists(triplet, model)

            if self._is_hit_at_n(triplet, dists, n=5):
                cum_hits_at_5 += 1

            if self._is_hit_at_n(triplet, dists, n=10):
                cum_hits_at_10 += 1

            if self._is_hit_at_n(triplet, dists, n=20):
                cum_hits_at_20 += 1

            cum_rank += float(self._triplet_rank(dists, raw_triplet.tail))

        hits_at_5 = cum_hits_at_5 / float(self._sample_size)
        hits_at_10 = cum_hits_at_10 / float(self._sample_size)
        hits_at_20 = cum_hits_at_20 / float(self._sample_size)
        mean_rank = cum_rank / float(self._sample_size)

        return MetricsBundle(
            mean_rank=mean_rank, 
            hits_at_5=hits_at_5, 
            hits_at_10=hits_at_10, 
            hits_at_20=hits_at_20,
        )

    def _corrupted_dists(self, triplet: torch.IntTensor, model: transh.TranshModel) -> List[float]:
        corrupted_triplets = torch.zeros(self._onto.entities_len(), 3, dtype=torch.int32)

        # Corrupting heads.
        for j in range(self._onto.entities_len()):
            corrupted_triplets[j, self._HEADS] = triplet[self._HEADS]
            corrupted_triplets[j, self._REL] = triplet[self._REL]
            corrupted_triplets[j, self._TAILS] = j
        
        scores = model(corrupted_triplets)
        return scores.flatten().tolist()

    def _is_hit_at_n(self, triplet: torch.IntTensor, dists: List[float], n: int) -> Tuple[float, float]:
        translations = self._onto.head_translations(triplet[self._HEADS])
        positive_tails = set()
        
        for trans in translations:
            if trans.rel == triplet[1]:
                positive_tails.add(trans.tail)
        
        for i in self._closest_triplets_indices(dists, n):
            if i in positive_tails:
                return True

        return False

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

    def _triplet_rank(self, dists: List[float], tail_idx: int) -> float:
        closer_count = 0
        
        for i, dist in enumerate(dists):
            if i == tail_idx:
                continue

            if dist < dists[tail_idx]:
                closer_count += 1

        return float(closer_count)
