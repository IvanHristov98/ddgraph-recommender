import math
import random
from typing import List, Tuple
import numpy as np

import ddgraph.graph as graph


class LatentGraph:
    entity_embeddings: List[np.ndarray]
    label_embeddings: List[np.ndarray]


def init_latent_graph(g: graph.UserItemGraph, dim_count: int) -> LatentGraph:
    lg = LatentGraph()
    lg.entity_embeddings = []
    
    for _ in range(len(g.user_entities) + len(g.item_entities)):
        lg.entity_embeddings.append(_init_embedding(dim_count))

    lg.label_embeddings = []

    for _ in g.relationships:
        embedding = _init_embedding(dim_count)
        length = np.linalg.norm(embedding)
        embedding /= length
        
        lg.label_embeddings.append(embedding)

    return lg


def train_minibatch(
    g: graph.UserItemGraph, 
    lg: LatentGraph, 
    minibatch_size: int, 
    margin: float = 1, 
    learning_rate: float = 0.01,
) -> LatentGraph:
    # Normalize all entity embeddings.
    for i in lg.entity_embeddings:
        length = np.linalg.norm(lg.entity_embeddings[i])
        lg.entity_embeddings[i] /= length

    # Sample a minibatch.
    minibatch = _triplets_minibatch(g.adj_list, minibatch_size)
    
    # Zip the minibatch triplets with corrupted counterparts.
    pair_minibatch = _pairs_minibatch(minibatch, lg)


def _init_embedding(dim_count: int) -> np.ndarray:
    return np.random.uniform(low=-6/math.sqrt(dim_count), high=6/math.sqrt(dim_count), size=dim_count)


# Note - we PROBABLY dont't care for any repetitions.
def _triplets_minibatch(adj_list: List[List[Tuple[int, int]]], size: int) -> List[Tuple[int, int, int]]:
    triplets = []
    
    for i in range(size):
        head_idx = random.randint(0, len(adj_list) - 1)
        
        # Hopefully we won't endup in an infinite cycle.
        if len(adj_list[head_idx]) == 0:
            continue
        
        label, tail_idx = random.randint(0, len(adj_list[head_idx]) - 1)
        
        triplets.append((head_idx, label, tail_idx))
    
    return triplets


def _pairs_minibatch(
    minibatch: List[Tuple[int, int, int]], 
    lg: LatentGraph,
) -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    pair_minibatch = []
    
    for h, l, t in minibatch:
        corrupted_entity_idx = random.randint(0, len(lg.entity_embeddings))

        # Toss the coin:
        # ---> Heads
        if random.randint(0, 1) == 0:
            pair_minibatch.append(((h, l, t), (corrupted_entity_idx, l, t)))
        # ---> Tails
        else:
            pair_minibatch.append(((h, l, t), (h, l, corrupted_entity_idx)))

    return pair_minibatch
