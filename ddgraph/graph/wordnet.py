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
    _rels_hpt: List[float]
    _rels_tph: List[float]

    def __init__(self, triplets: List[onto.Triplet], relationships: List[str], entities: List[str]) -> None:
        super().__init__()
        
        self._triplets = triplets
        self._relationships = relationships
        self._entities = entities
        
        # done as an optimisation for later searches
        self._reorder_triplets()

        self._rels_tph = self._find_rels_tph()
        self._rels_hpt = self._find_rels_hpt()

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

    # TODO: Fix code repetition
    def corruption_probs(self, rel_idx: int) -> onto.CorruptionProbs:
        tph = self._rels_tph[rel_idx]
        hpt = self._rels_hpt[rel_idx]

        return onto.CorruptionProbs(
            head=(tph/(tph + hpt)),
            tail=(hpt/(tph + hpt))
        )

    def entities_len(self) -> int:
        return len(self._entities)

    def relations_len(self) -> int:
        return len(self._relationships)

    def head_translations(self, head_idx: int) -> List[onto.Trans]:
        start, end = self._head_ranges[head_idx]
        translations = []
        
        for triplet in self._triplets[start:end]:
            translations.append(onto.Trans(rel=triplet.rel, tail=triplet.tail))
        
        return translations

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

    def _find_rels_tph(self) -> List[float]:
        rels_tph = [0] * len(self._relationships)
        
        for i, _ in enumerate(self._relationships):
            rels_tph[i] = self._find_mean_tph(i)
        
        return rels_tph

    def _find_mean_tph(self, rel: int) -> float:
        tail_counts = [0] * len(self._entities)
        
        for _, triplet in enumerate(self._triplets):
            if triplet.rel == rel:
                tail_counts[triplet.head] += 1

        return self._mean_of_observations(tail_counts)

    def _find_rels_hpt(self) -> List[float]:
        rels_hpt = [0] * len(self._relationships)
        
        for i, _ in enumerate(self._relationships):
            rels_hpt[i] = self._find_mean_hpt(i)

        return rels_hpt

    def _find_mean_hpt(self, rel: int) -> float:
        head_counts = [0] * len(self._entities)
        
        for _, triplet in enumerate(self._triplets):
            if triplet.rel == rel:
                head_counts[triplet.tail] += 1

        return self._mean_of_observations(head_counts)

    def _mean_of_observations(self, observations: List[int]) -> float:
        total_observations = 0
        num_entities_observed = 0
        
        for cnt in observations:
            total_observations += cnt
            
            if cnt > 0:
                num_entities_observed += 1

        return total_observations / num_entities_observed


class WNParser:
    _data_path: Path
    _file_name: str

    def __init__(self, data_path: Path, file_name: str) -> None:
        self._data_path = data_path
        self._file_name = file_name

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
        return Path(self._data_path, self._file_name)
