import math
import logging
import random
from typing import Tuple

import torch
import torch.utils.data as torch_data

import ddgraph.graph as graph


class TranseModel(torch.nn.Module):
    def __init__(self, num_entities: int, num_rels: int, k: int):
        super(TranseModel, self).__init__()

        self.entity_embeddings = self._entity_embeddings(num_entities, k)
        self.rel_embeddings = self._rel_embeddings(num_rels, k)

    def forward(self, x):
        head_emb = self.entity_embeddings(x[:, 0])
        rel_emb = self.rel_embeddings(x[:, 1])
        tail_emb = self.entity_embeddings(x[:, 2])

        return (head_emb + rel_emb - tail_emb).pow(2).sum(1).sqrt()

    def entity_dist(self, a: int, b: int) -> float:
        a_emb = self.entity_embeddings(torch.tensor(a))
        b_emb = self.entity_embeddings(torch.tensor(b))

        return (a_emb - b_emb).pow(2).sum(0).sqrt().item()

    def _entity_embeddings(self, num_entities: int, k: int) -> torch.nn.Embedding:
        high, low = self._initial_boundaries(k)
        entity_tensor = (high - low) * torch.rand(num_entities, k) + low

        # Normalize entity embeddings to prevent a trivial optimisation of the loss function.
        return torch.nn.Embedding.from_pretrained(entity_tensor, freeze=False, max_norm=1)

    def _rel_embeddings(self, num_rels: int, k: int) -> torch.nn.Embedding:
        high, low = self._initial_boundaries(k)
        rel_tensor = (high - low) * torch.rand(num_rels, k) + low
        
        for _, rel_emb in enumerate(rel_tensor):
            rel_emb /= torch.linalg.norm(rel_emb, ord=2)
        
        return torch.nn.Embedding.from_pretrained(rel_tensor, freeze=False)

    def _initial_boundaries(self, k: int) -> Tuple[float, float]:
        return -6.0 / math.sqrt(k), 6.0 / math.sqrt(k)


class Trainer:
    _training_loader: torch_data.DataLoader
    _onto: graph.Ontology
    _optimizer: torch.optim.Optimizer
    _model: TranseModel
    _margin: float
    _epoch: int

    def __init__(
        self, 
        training_loader: torch_data.DataLoader,
        onto: graph.Ontology,
        optimizer: torch.optim.Optimizer,
        model: TranseModel,
        margin: float,
    ) -> None:
        self._training_loader = training_loader
        self._onto = onto
        self._optimizer = optimizer
        self._model = model
        self._margin = margin

        self._epoch = 0

    def train_one_epoch(self) -> None:
        self._epoch += 1
        minibatch_count = 0
        cum_loss = 0

        # Sample a minibatch of triplets on each turn.
        for _, triplets in enumerate(self._training_loader):
            self._optimizer.zero_grad()
            
            corrupted_triplets = corrupted_counterparts(self._onto, triplets)

            out = self._model(triplets)
            corrupted_out = self._model(corrupted_triplets)
            
            loss = torch.nn.functional.relu(self._margin + out - corrupted_out).sum()
            loss.backward()
            
            # Adjust learning weights
            self._optimizer.step()

            cum_loss += loss / len(triplets)
            minibatch_count += 1
        
        logging.info(f"[Epoch {self._epoch}] Average loss ---> {cum_loss / minibatch_count}")

    def epoch(self) -> int:
        return self._epoch

    def model(self) -> TranseModel:
        return self._model


def corrupted_counterparts(onto: graph.Ontology, triplets: torch.IntTensor) -> torch.IntTensor:
    corrupted_triplets = torch.clone(triplets)
        
    for _, triplet in enumerate(corrupted_triplets):
        _corrupt(onto, triplet)

    return corrupted_triplets


def _corrupt(onto: graph.Ontology, triplet: torch.IntTensor) -> None:
    corrupted_entity_idx = random.randint(0, onto.entities_len() - 1)

    # To pick a triplet side toss the coin:
    # ---> Heads
    if random.randint(0, 1) == 0:
        triplet[0] = corrupted_entity_idx
    # ---> Tails
    else:
        triplet[2] = corrupted_entity_idx
