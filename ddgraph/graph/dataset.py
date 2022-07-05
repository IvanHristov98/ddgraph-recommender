import torch
import torch.utils.data as torch_data

import ddgraph.graph.ontology as onto


class TripletDataset(torch_data.Dataset):
    _onto: onto.Ontology

    def __init__(self, onto: onto.Ontology) -> None:
        self._onto = onto

    def __len__(self) -> int:
        return self._onto.triplets_len()

    def __getitem__(self, idx):
        triplet = self._onto.get_triplet(idx)
        
        return tensorify_triplet(triplet)


def tensorify_triplet(triplet: onto.Triplet) -> torch.IntTensor:
    return torch.tensor([triplet.head, triplet.rel, triplet.tail])

def untensorify_triplet(triplet: torch.IntTensor) -> onto.Triplet:
    return onto.Triplet(head=triplet[0], rel=triplet[1], tail=triplet[2])
