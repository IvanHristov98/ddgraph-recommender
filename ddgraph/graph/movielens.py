import re
from pathlib import Path
from typing import List, Set, Tuple

import ddgraph.graph.ontology as onto


LIKES = "likes"
DISLIKES = "dislikes"


class MLRecOntology(onto.RecOntology):
    # Adjacency list is needed for QPCS.
    _adj_list: List[List[onto.Trans]]
    # _neighbour_counts has the number encountered of dests in _adj_list until a head is encountered.
    # It has a len of len(adj_list) + 1 where the last item contains the total number of triplets.
    # It is used for faster indexing (O(log n) because of binary search) and triplet counting (O(1)).
    _neighbour_counts: List[int]
    _relationships: List[str]
    _user_entities: List[str]
    _item_entities: List[str]
    _rels_hpt: List[float]
    _rels_tph: List[float]

    def __init__(
        self, 
        adj_list: List[List[onto.Trans]], 
        relationships: List[str], 
        user_entities: List[str], 
        item_entities: List[str],
    ) -> None:
        super().__init__()

        self._adj_list = adj_list
        self._relationships = relationships
        self._user_entities = user_entities
        self._item_entities = item_entities
        
        self._neighbour_counts = self._get_neighbour_counts(adj_list)
        
        self._rels_tph = self._find_rels_tph()
        self._rels_hpt = self._find_rels_hpt()

    def exists(self, triplet: onto.Triplet) -> bool:
        if not self._head_exists(triplet.head):
            return False

        for neighbour in self._adj_list[triplet.head]:
            rel, tail = neighbour

            if rel == triplet.rel and tail == triplet.tail:
                return True

        return False

    def add_triplets(self, triplets: List[onto.Triplet]) -> None:
        for _, triplet in enumerate(triplets):
            if not self._head_exists(triplet.head):
                raise onto.TripletOutOfBoundsError(f"non-existing head {triplet.head}")
            
            if not self._rel_exists(triplet.rel):
                raise onto.TripletOutOfBoundsError(f"non-existing rel {triplet.rel}")
        
            if not self._item_exists(triplet.tail):
                raise onto.TripletOutOfBoundsError(f"non-existing tail {triplet.rel}")

            self._adj_list[triplet.head].append(onto.Trans(triplet.rel, triplet.tail))

        # Update the length of the graph
        self._neighbour_counts = self._get_neighbour_counts(self._adj_list)

    def triplets_len(self) -> int:
        return self._neighbour_counts[len(self._neighbour_counts) - 1]

    def get_triplet(self, triplet_idx: int) -> onto.Triplet:
        head = self._head_for_triplet_at(triplet_idx)

        neigbours = self._adj_list[head]
        neighbour_idx = triplet_idx - (self._neighbour_counts[head] - len(neigbours))

        trans = neigbours[neighbour_idx]

        return onto.Triplet(head=head, rel=trans.rel, tail=trans.tail)

    def corruption_probs(self, rel_idx: int) -> onto.CorruptionProbs:
        tph = self._rels_tph[rel_idx]
        hpt = self._rels_hpt[rel_idx]
        
        return onto.CorruptionProbs(
            head=(tph/(tph + hpt)),
            tail=(hpt/(tph + hpt))
        )

    def entities_len(self) -> int:
        return len(self._user_entities) + len(self._item_entities)

    def relations_len(self) -> int:
        return len(self._relationships)

    def head_translations(self, head_idx: int) -> List[onto.Trans]:
        return self._adj_list[head_idx]

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

    def _get_neighbour_counts(self, adj_list: List[List[Tuple[int, int]]]) -> List[int]:
        total_triplets = 0
        neighbour_counts = []

        for head in range(len(adj_list)):
            total_triplets += len(adj_list[head])
            neighbour_counts.append(total_triplets)

        return neighbour_counts

    def _head_exists(self, idx: int) -> bool:
        return idx >= 0 and idx < len(self._adj_list)

    def _rel_exists(self, idx: int) -> bool:
        return idx >= 0 and idx < len(self._relationships)

    def _item_exists(self, idx: int) -> bool:
        return idx >= len(self._user_entities) and idx < len(self._user_entities) + len(self._item_entities)

    def _triplet_potentially_exists(self, idx: int) -> bool:
        return idx >= 0 and idx < self.triplets_len()

    def _head_for_triplet_at(self, idx: int) -> int:
        left = 0
        right = len(self._neighbour_counts) - 1
        
        if not self._triplet_potentially_exists(idx):
            raise onto.TripletOutOfBoundsError(f"invalid head and idx {idx} during triplet search")

        # Done because every item of _neighbour_counts contains the number of
        # triplets for the given head.
        idx += 1
        
        while left < right:
            mid = left + (right - left) // 2

            if self._neighbour_counts[mid] == idx and mid > 0 and self._neighbour_counts[mid - 1] < idx:
                return mid
            elif self._neighbour_counts[mid] < idx:
                left = mid + 1
            else:
                right = mid
        
        # left can never become greater than right and we have a sparse representation where
        # an idx of a triplet would most probably not be found in an array but is a valid idx.
        return left

    def _find_rels_tph(self) -> List[float]:
        rels_tph = [0] * len(self._relationships)
        
        for i, _ in enumerate(self._relationships):
            rels_tph[i] = self._find_mean_tph(i)
        
        return rels_tph

    def _find_mean_tph(self, rel: int) -> float:
        tail_counts = [0] * len(self._user_entities)
        
        for head, _ in enumerate(self._adj_list):
            for _, trans in enumerate(self._adj_list[head]):
                if trans.rel == rel:
                    tail_counts[head] += 1
    
        return self._mean_of_observations(tail_counts)

    def _find_rels_hpt(self) -> List[float]:
        rels_hpt = [0] * len(self._relationships)
        
        for i, _ in enumerate(self._relationships):
            rels_hpt[i] = self._find_mean_hpt(i)

        return rels_hpt

    def _find_mean_hpt(self, rel: int) -> float:
        head_counts = [0] * len(self._item_entities)
        
        for head, _ in enumerate(self._adj_list):
            for _, trans in enumerate(self._adj_list[head]):
                if trans.rel == rel:
                    head_counts[trans.tail - len(self._user_entities)] += 1

        return self._mean_of_observations(head_counts)
    
    def _mean_of_observations(self, observations: List[int]) -> float:
        total_observations = 0
        num_entities_observed = 0
        
        for cnt in observations:
            total_observations += cnt
            
            if cnt > 0:
                num_entities_observed += 1

        return total_observations / num_entities_observed


class MovieLensParser:
    LIKES_IDX = 0
    DISLIKES_IDX = 1
    
    _UDATA_SEPARATOR = " "
    _PIPE_SPLITTER = "|"

    _UDATA_USER_ID_POS = 0
    _UDATA_ITEM_ID_POS = 1
    _UDATA_RATING_POS = 2

    _UUSER_USER_ID_POS = 0
    
    _UITEM_ITEM_ID_POS = 0

    _EXPECTED_UDATA_CHUNKS_PER_LINE = 4
    _EXPECTED_UUSER_CHUNKS_PER_LINE = 5
    _EXPECTED_UITEM_CHUNKS_PER_LINE = 24
    
    _data_path: Path
    _relationships: List[str]
    _file_name: str

    def __init__(self, data_path: Path, file_name: str) -> None:
        self._data_path = data_path
        self._relationships = [LIKES, DISLIKES]
        self._file_name = file_name

    def parse(self) -> MLRecOntology:
        user_entities = self._user_entities()
        item_entities = self._item_entities()
        
        adj_list = self._adj_list(len(user_entities))

        return MLRecOntology(adj_list, self._relationships, user_entities, item_entities)

    def _user_entities(self) -> str:
        entities = []
        
        for uuser_line in self._uuser().split("\n"):
            chunks = uuser_line.split(self._PIPE_SPLITTER)
            
            if len(chunks) != self._EXPECTED_UUSER_CHUNKS_PER_LINE:
                continue

            entities.append(f"{chunks[1]}-{chunks[2]}-{chunks[3]}-{chunks[4]}")
        
        return entities

    def _uuser(self) -> str:
        return self._read_file(self._uuser_path())

    def _uuser_path(self) -> Path:
        return Path(self._data_path, "u.user")

    def _item_entities(self) -> str:
        entities = []
        
        for uitem_line in self._uitem().split("\n"):
            chunks = uitem_line.split(self._PIPE_SPLITTER)

            if len(chunks) != self._EXPECTED_UITEM_CHUNKS_PER_LINE:
                continue

            entities.append(chunks[1])
        
        return entities

    def _uitem(self) -> str:
        return self._read_file(self._uitem_path())

    def _uitem_path(self) -> Path:
        return Path(self._data_path, "u.item")

    def _udata(self) -> str:        
        udata = self._read_file(self._udata_path())

        return re.sub(r"[^\S\n\r]+", self._UDATA_SEPARATOR, udata)

    def _read_file(self, path: Path) -> str:    
        with open(path, encoding="latin-1") as stream:
            return stream.read()

    def _udata_path(self) -> Path:
        return Path(self._data_path, self._file_name)

    def _adj_list(self, user_count: int) -> List[List[onto.Trans]]:        
        udata = self._udata()
        adj_list = []

        # TODO: Find better syntactic sugar.
        for _ in range(user_count):
            adj_list.append([])

        relationship_indices = {LIKES: 0, DISLIKES: 1}

        for udata_line in udata.split("\n"):
            chunks = udata_line.split(self._UDATA_SEPARATOR)

            if len(chunks) != self._EXPECTED_UDATA_CHUNKS_PER_LINE:
                break

            # Indices start from 1 in the dataset but we translate them
            # to an ordering suitable for a normal programming language.
            user_id = int(chunks[self._UDATA_USER_ID_POS]) - 1
            item_id = user_count + int(chunks[self._UDATA_ITEM_ID_POS]) - 1
            rating = int(chunks[self._UDATA_RATING_POS])
            label = self._rating_to_label(rating)
            
            adj_list[user_id].append(onto.Trans(rel=relationship_indices[label], tail=item_id))

        return adj_list

    def _rating_to_label(self, rating: int) -> str:
        if rating >= 4:
            return LIKES

        return DISLIKES
