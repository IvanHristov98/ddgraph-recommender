from typing import Dict, List

import torch

import ddgraph.graph as graph
import ddgraph.graph.movielens as mvlens
import ddgraph.transh as transh
import ddgraph.selector as selector


class DDGraphTrainer:
    _TRANSH_EPOCHS_PER_EPOCH = 10
    
    _dataset: graph.TripletDataset
    _transh_trainer: transh.Trainer
    _qpcs: selector.QPCSelector
    
    def __init__(self, dataset: graph.TripletDataset, transh_trainer: transh.Trainer, qpcs: selector.QPCSelector) -> None:
        self._dataset = dataset
        self._transh_trainer = transh_trainer
        self._qpcs = qpcs

    # TODO: Replace epochs with a metric-based stop criteria.
    def train(self, epochs: int = 1) -> None:
        for _ in range(epochs):
            self._train_one_epoch()

    def _train_one_epoch(self) -> None:
        for _ in range(self._TRANSH_EPOCHS_PER_EPOCH):
            self._transh_trainer.train_one_epoch()

        neighbours = self._qpcs.select_items()
        
        # Add the newly found neighbours to the graph
        self._update_graph(neighbours)

    def _update_graph(self, neighbours: Dict[int, List[int]]) -> None:
        triplets = []

        for user, items in neighbours.items():            
            for item in items:
                triplets.append([user, mvlens.MovieLensParser.LIKES_IDX, item])

        self._dataset.add_triplets(torch.tensor(triplets))
