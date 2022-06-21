import re
from pathlib import Path
from typing import List, Tuple
from urllib.parse import uses_fragment

import ddgraph.graph.graph as graph


LIKES = "likes"
DISLIKES = "dislikes"


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

    def __init__(self, data_path: Path) -> None:
        self._data_path = data_path

    def parse(self) -> graph.UserItemGraph:
        g = graph.UserItemGraph()
        g.relationships = [LIKES, DISLIKES]
        g.user_entities = self._user_entities()
        g.item_entities = self._item_entities()
        
        g.adj_list = self._adj_list(len(g.user_entities))

        return g

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

    def _uuser_path(self) -> str:
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

    def _uitem_path(self) -> str:
        return Path(self._data_path, "u.item")

    def _udata(self) -> str:        
        udata = self._read_file(self._udata_path())

        return re.sub(r"[^\S\n\r]+", self._UDATA_SEPARATOR, udata)

    def _read_file(self, path: Path) -> str:    
        with open(path, encoding="latin-1") as stream:
            return stream.read()

    def _udata_path(self) -> Path:
        return Path(self._data_path, "u.data")

    def _adj_list(self, user_count: int) -> List[List[Tuple[int, int]]]:        
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
            
            adj_list[user_id].append((relationship_indices[label], item_id))

        return adj_list

    def _rating_to_label(self, rating: int) -> str:
        if rating >= 4:
            return LIKES

        return DISLIKES
