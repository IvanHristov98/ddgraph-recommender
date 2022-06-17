from typing import NamedTuple

import numpy as np


class User(NamedTuple):
    metadata: str


class Item(NamedTuple):
    name: str


# class EntityEmbeddings:
#     def __init__(self, user_embeddings, item_embeddings) -> None:
#         self._user_count = len(user_embeddings)
#         self._embeddings = user_embeddings + item_embeddings
        
#         print(self._embeddings)
