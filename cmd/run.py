from os import environ
from pathlib import Path
import torch

import torch.utils.data as torch_data

import ddgraph.graph as graph
import ddgraph.transe as transe


class Config:
    data_dir: Path

    def ml_100k_dir(self) -> Path:
        return Path(self.data_dir, "ml-100k")


def main():
    cfg = _config()
    parser = graph.MovieLensParser(cfg.ml_100k_dir())

    g = parser.parse()
    
    train_dataloader = torch_data.DataLoader(g, batch_size=64, shuffle=True)
    
    train_features = next(iter(train_dataloader))
    
    model = transe.TranseModel(num_entities=5, num_rels=5, k=3)
    print(model)

    dist = model(torch.tensor([0, 2, 4])).item()
    print(dist)


def _config() -> Config:
    cfg = Config()
    cfg.data_dir = environ.get("DATA_DIR", "")
    
    return cfg


if __name__ == "__main__":
    main()
