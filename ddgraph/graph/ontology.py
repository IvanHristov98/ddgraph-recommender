import abc
from typing import List, NamedTuple, Set


class TripletOutOfBoundsError(Exception):
    """
    Thrown whenever a triplet is wrongly indexed.
    """


class Triplet(NamedTuple):
    head: int
    rel: int
    tail: int


class Trans(NamedTuple):
    rel: int
    tail: int


class CorruptionProbs(NamedTuple):
    head: float
    tail: float


class Ontology(abc.ABC):
    def __init__(self) -> None:
        pass

    def exists(self, triplet: Triplet) -> bool:
        raise NotImplementedError('')

    def triplets_len(self) -> int:
        raise NotImplementedError('')

    def get_triplet(self, triplet_idx: int) -> Triplet:
        raise NotImplementedError('')

    def corruption_probs(self, rel_idx: int) -> CorruptionProbs:
        raise NotImplementedError('')

    def entities_len(self) -> int:
        raise NotImplementedError('')

    def relations_len(self) -> int:
        raise NotImplementedError('')


class RecOntology(Ontology):
    def __init__(self) -> None:
        pass

    def add_triplets(self, triplets: List[Triplet]) -> None:
        raise NotImplementedError('')
    
    def item_indices(self) -> Set[int]:
        raise NotImplementedError('')
    
    def user_indices(self) -> Set[int]:
        raise NotImplementedError('')
