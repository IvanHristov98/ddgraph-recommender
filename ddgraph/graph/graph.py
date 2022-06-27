import random
from typing import List, Set, Tuple
from numpy import indices

import torch
import torch.utils.data as torch_data


class TripletOutBoundsError(Exception):
    """
    Thrown whenever a triplet is wrongly indexed.
    """


class TripletDataset(torch_data.Dataset):
    # Adjacency list is needed for QPCS.
    _adj_list: List[List[Tuple[int, int]]]
    # _neighbour_counts has the number encountered of dests in _adj_list until a head is encountered.
    # It has a len of len(adj_list) + 1 where the last item contains the total number of triplets.
    # It is used for faster indexing (O(log n) because of binary search) and triplet counting (O(1)).
    _neighbour_counts: List[int]
    _relationships: List[str]
    _user_entities: List[str]
    _item_entities: List[str]

    def __init__(
        self, 
        adj_list: List[List[Tuple[int, int]]], 
        relationships: List[str], 
        user_entities: List[str], 
        item_entities: List[str],
    ) -> None:
        self._adj_list = adj_list
        self._relationships = relationships
        self._user_entities = user_entities
        self._item_entities = item_entities
        
        self._neighbour_counts = self._get_neighbour_counts(adj_list)

    def __len__(self) -> int:
        return self._triplets_count()

    # O(log(n))
    def __getitem__(self, idx):
        head = self._head_for_triplet_at(idx)

        neigbours = self._adj_list[head]
        neighbour_idx = idx - (self._neighbour_counts[head] - len(neigbours))

        rel, tail = neigbours[neighbour_idx]
        
        # the data points are triplets of <head, relationship, tail>
        return torch.tensor([head, rel, tail])

    def corrupted_counterparts(self, triplets: torch.IntTensor) -> torch.IntTensor:
        corrupted_triplets = torch.clone(triplets)
        
        for _, triplet in enumerate(corrupted_triplets):
            self._corrupt(triplet)

        return corrupted_triplets

    def exists(self, triplet: torch.IntTensor) -> bool:
        if triplet[0] < 0 or triplet[0] > len(self._adj_list) - 1:
            return False

        for neighbour in self._adj_list[triplet[0]]:
            rel, tail = neighbour

            if rel == triplet[1] and tail == triplet[2]:
                return True

        return False

    def entities_len(self) -> int:
        return len(self._user_entities) + len(self._item_entities)

    def item_indices(self) -> Set[int]:
        indices = set()
        
        for i in range(len(self._user_entities), self.entities_len()):
            indices.add(i)

        return indices

    def user_indices(self) -> Set[int]:
        indices = set()
        
        for i in range(len(self._user_entities)):
            indices.add(i)

        return indices

    def rel_len(self) -> int:
        return len(self._relationships)

    def user_neighbours(self, user: int) -> List[Tuple[int, int]]:
        return self._adj_list[user]

    def _corrupt(self, triplet: torch.IntTensor) -> None:
        # To reduce bias toss the coin:
        # ---> User
        if random.randint(0, 1) == 0:
            corrupted_entity_idx = random.randint(0, len(self._user_entities) - 1)
        # ---> Item
        else:
            corrupted_entity_idx = random.randint(len(self._user_entities), len(self._user_entities) + len(self._item_entities) - 1)

        # To pick a triplet side toss the coin:
        # ---> Heads
        if random.randint(0, 1) == 0:
            triplet[0] = corrupted_entity_idx
        # ---> Tails
        else:
            triplet[2] = corrupted_entity_idx

    def _get_neighbour_counts(self, adj_list: List[List[Tuple[int, int]]]) -> List[int]:
        total_triplets = 0
        neighbour_counts = []

        for head in range(len(adj_list)):
            total_triplets += len(adj_list[head])
            neighbour_counts.append(total_triplets)

        return neighbour_counts

    def _head_for_triplet_at(self, idx: int) -> int:
        left = 0
        right = len(self._neighbour_counts) - 1
        
        if idx < 0 or self._neighbour_counts[right] - 1 < idx:
            raise TripletOutBoundsError(f"Expected triplet idx {idx} to be between 0 and {self._neighbour_counts[right] - 1} inclusively.")
        
        # Done because every item of _neighbour_counts contains the number of
        # triplets for the given head.
        idx += 1
        
        while left < right:
            mid = left + (right - left) // 2

            if self._neighbour_counts[mid] == idx:
                return mid
            elif self._neighbour_counts[mid] < idx:
                left = mid + 1
            else:
                right = mid
        
        # left can never become greater than right and we have a sparse representation where
        # an idx of a triplet would most probably not be found in an array but is a valid idx.
        return left

    def _triplets_count(self) -> int:
        return self._neighbour_counts[len(self._neighbour_counts) - 1]
