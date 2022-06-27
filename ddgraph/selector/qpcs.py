from typing import Dict, List, Set, Tuple

import torch

import ddgraph.transe as transe
import ddgraph.graph as graph
import ddgraph.graph.movielens as mvlens


# QPCSelector stands for quantile progressive candidate selector
# The class is using the naming conventions from the paper.
class QPCSelector:
    _dataset: graph.TripletDataset
    _model: transe.TranseModel
    _num_quantiles: int
    # should be a real number between 0 and 1
    _relevance_bias: float
    
    def __init__(
        self, 
        dataset: graph.TripletDataset,
        model: transe.TranseModel, 
        num_quantiles: int = 4, 
        relevance_bias: float = 0.5,
    ) -> None:
        self._model = model
        self._dataset = dataset
        self._num_quantiles = num_quantiles
        self._relevance_bias = relevance_bias

    # Returns Dict[user -> diversified_items]
    def select_items(self) -> Dict[int, List[int]]:
        users = self._dataset.user_indices()
        neighbours = dict()
        
        for user in users:
            nu = self._select_items_for_user(user)
            neighbours[user] = nu
        
        return neighbours

    @torch.no_grad()
    # TODO: Implement with more epochs.
    def _select_items_for_user(self, user: int) -> List[int]:
        pu = self._interacted_items(user)
        cu = self._non_interacted_items(pu)
        nu = set()
        
        weighted_item_sets = self._split_item_space_by_quantiles(pu, cu, nu)
        
        for weighted_item_set in weighted_item_sets:
            raw_triplets = []
            
            for weighted_item in weighted_item_set:
                raw_triplets.append([user, mvlens.MovieLensParser.LIKES_IDX, weighted_item[0]])
            
            triplets = torch.tensor(raw_triplets)
            relevances = self._model(triplets)

            max_score = 0
            best_item = -1
            
            for k, weighted_item in enumerate(weighted_item_set):
                # (1 - bias) * Dist + bias * Relevance
                score = (1 - self._relevance_bias) * weighted_item[1] + self._relevance_bias * relevances[k]
                
                if max_score < score:
                    max_score = score
                    best_item = weighted_item[0]
            
            nu.add(best_item)
            cu.remove(best_item)

        return list(nu)

    def _interacted_items(self, user: int) -> Set[int]:
        neighbours = self._dataset.user_neighbours(user)
        neighbours = list(filter(lambda neighbour: neighbour[0] == mvlens.MovieLensParser.LIKES_IDX, neighbours))
        
        interacted_items = set()
        
        for neighbour in neighbours:
            interacted_items.add(neighbour[1])
        
        return interacted_items

    def _non_interacted_items(self, interacted_items: Set[int]) -> Set[int]:
        non_interacted_items = self._dataset.item_indices()
        non_interacted_items = non_interacted_items.difference(interacted_items)
        
        return non_interacted_items

    def _split_item_space_by_quantiles(self, pu: Set[int], cu: Set[int], nu: Set[int]) -> List[List[Tuple[int, float]]]:
        dists = self._find_min_dists(pu, cu, nu)
        return self._split_space_from_dists(dists)

    def _split_space_from_dists(self, dists: List[Tuple[int, float]]) -> List[List[Tuple[int, float]]]:
        sorted_dists = sorted(dists, key=lambda x: x[1])
        
        splitter_indices = []
        
        for i in range(self._num_quantiles):
            splitter_indices.append((len(sorted_dists) * (i + 1)) // self._num_quantiles)
    
        item_sets = []
        prev_split_idx = 0
    
        for splitter_idx in splitter_indices:
            item_set = []
            
            # Copy all items to item set
            for i in range(prev_split_idx, splitter_idx):
                item_set.append(sorted_dists[i])

            item_sets.append(item_set)
            prev_split_idx = splitter_idx 
        
        return item_sets

    def _find_min_dists(self, pu: Set[int], cu: Set[int], nu: Set[int]) -> List[Tuple[int, float]]:
        reachable_items = pu.union(nu)
        dists = []
        
        for item in cu:
            min_dist = 0

            for i, reachable_item in enumerate(reachable_items):
                dist = self._model.entity_dist(item, reachable_item)
                
                if min_dist > dist or i == 0:
                    min_dist = dist
            
            dists.append((item, min_dist))
        
        return dists
