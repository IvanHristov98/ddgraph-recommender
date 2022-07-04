from pathlib import Path
from typing import List, Tuple

from torch import chunk

import ddgraph.graph.ontology as onto


class WNOntology(onto.Ontology):
    _triplets: List[onto.Triplet]
    # head ranges frpm [triplet; triplet)
    _head_ranges: List[Tuple[int, int]]
    _relationships: List[str]
    _entities: List[str]

    def __init__(self, triplets: List[onto.Triplet], relationships: List[str], entities: List[str]) -> None:
        super().__init__()
        
        self._triplets = triplets
        self._relationships = relationships
        self._entities = entities
        
        # done as an optimisation for later searches
        self._reorder_triplets()

    def exists(self, triplet: onto.Triplet) -> bool:     
        start, end = self._head_ranges[triplet.head]

        for i in range(start, end):        
            triplet = self._triplets[i]
            
            if triplet.rel == triplet.rel and triplet.tail == triplet.tail:
                return True
        
        return False

    def triplets_len(self) -> int:
        return len(self._triplets)

    def get_triplet(self, triplet_idx: int) -> onto.Triplet:
        return self._triplets[triplet_idx]

    def entities_len(self) -> int:
        return len(self._entities)

    def relations_len(self) -> int:
        return len(self._relationships)

    def _reorder_triplets(self) -> None:
        self._triplets = sorted(self._triplets, key=lambda x: x.head)
    
        self._head_ranges = [None] * len(self._entities)
        curr_triplet_idx = 0
        
        for head in range(len(self._head_ranges)):
            start = curr_triplet_idx
            
            while True:
                if curr_triplet_idx < len(self._triplets) and self._triplets[curr_triplet_idx].head == head:
                    curr_triplet_idx += 1
                else:
                    break

            self._head_ranges[head] = (start, curr_triplet_idx)

        if curr_triplet_idx != len(self._triplets):
            raise Exception("unexhausted triplets during counting per head")


class WNParser:
    _data_path: Path
    
    def __init__(self, data_path: Path) -> None:
        self._data_path = data_path

    def parse(self) -> WNOntology:
        with open(self._train_path()) as stream:
            lines = stream.readlines()

        entities = set()
        rels = set()

        for i, line in enumerate(lines):
            chunks = line.rstrip().split("\t")
            
            entities.add(chunks[0])
            entities.add(chunks[1])
            rels.add(chunks[2])
        
        entities = list(entities)
        rels = list(rels)
        
        rev_entities = dict()
        rev_rels = dict()
        
        for i, ent in enumerate(entities):
            rev_entities[ent] = i
    
        for i, rel in enumerate(rels):
            rev_rels[rel] = i
        
        triplets = [None] * len(lines)
        
        for i, line in enumerate(lines):
            chunks = line.rstrip().split("\t")

            head = rev_entities[chunks[0]]
            tail = rev_entities[chunks[1]]
            label = rev_rels[chunks[2]]

            triplet = onto.Triplet(head=head, rel=label, tail=tail)
            triplets[i] = triplet
        
        return WNOntology(triplets, rels, entities)

    def _train_path(self) -> Path:
        return Path(self._data_path, "train.txt")
