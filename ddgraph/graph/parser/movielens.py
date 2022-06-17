import chunk
import re
from pathlib import Path
from typing import List, NamedTuple, Tuple


LIKES = "likes"
DISLIKES = "dislikes"


class UserItemGraph:
    # item offset is the adj_list length
    adj_list: List[List[Tuple[int, int]]]
    relationships: List[str]


class MovieLensParser:
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
    # The movielens 100k dataset has distinct orders for users and items
    # but a way is needed to index them in an adjacency matrix and other graph structures.
    # Thus item_id_offset pushes all item ids after user ids
    _item_id_offset: int

    def __init__(self, data_path: Path) -> None:
        self._data_path = data_path
        self._item_id_offset = 0

    def parse(self) -> UserItemGraph:    
        self._item_id_offset = self._num_users()

        udata = self._udata()
        return self._user_item_graph(udata)

    def _uuser(self) -> str:
        return self._read_file(self._uuser_path())

    def _uuser_path(self) -> str:
        return Path(self._data_path, "u.user")

    def _udata(self) -> str:        
        udata = self._read_file(self._udata_path())

        return re.sub(r"[^\S\n\r]+", self._UDATA_SEPARATOR, udata)

    def _read_file(self, path: Path) -> str:    
        with open(path) as stream:
            return stream.read()

    def _udata_path(self) -> Path:
        return Path(self._data_path, "u.data")

    def _num_users(self) -> int:
        offset = 0
        
        for uuser_line in self._uuser().split("\n"):
            chunks = uuser_line.split(self._PIPE_SPLITTER)
            
            if len(chunks) != self._EXPECTED_UUSER_CHUNKS_PER_LINE:
                break
            
            offset += 1
        
        return offset

    def _user_item_graph(self, udata: str) -> UserItemGraph:        
        graph = UserItemGraph()
        graph.relationships = [LIKES, DISLIKES]
        graph.adj_list = []

        # TODO: Find better syntactic sugar.
        for _ in range(self._item_id_offset):
            graph.adj_list.append([])

        relationship_indices = {LIKES: 0, DISLIKES: 1}

        for udata_line in udata.split("\n"):
            chunks = udata_line.split(self._UDATA_SEPARATOR)

            if len(chunks) != self._EXPECTED_UDATA_CHUNKS_PER_LINE:
                break

            # Indices start from 1 in the dataset but we translate them
            # to an ordering suitable for a normal programming language.
            user_id = int(chunks[self._UDATA_USER_ID_POS]) - 1
            item_id = self._item_id_offset + int(chunks[self._UDATA_ITEM_ID_POS]) - 1
            rating = int(chunks[self._UDATA_RATING_POS])
            label = self._rating_to_label(rating)
            
            graph.adj_list[user_id].append((relationship_indices[label], item_id))

        return graph

    def _rating_to_label(self, rating: int) -> str:
        if rating >= 4:
            return LIKES

        return DISLIKES
