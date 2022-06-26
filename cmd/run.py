from os import environ
from pathlib import Path

import torch.utils.data as torch_data

import ddgraph.graph as graph


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
    
    print(f"Feature batch shape: {train_features.size()}")
    # print(f"Lables batch shape: {train_labels.size()}")
    
    print(train_features[0])
    
    corrupted_triplet = g.corrupted_counterpart(train_features[0])
    print(corrupted_triplet)


def _config() -> Config:
    cfg = Config()
    cfg.data_dir = environ.get("DATA_DIR", "")
    
    return cfg


if __name__ == "__main__":
    main()
