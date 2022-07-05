import logging
from os import environ
from pathlib import Path
import torch

import torch.utils.data as torch_data

import ddgraph.graph as graph
import ddgraph.transh as transh
import ddgraph.selector as selector
import ddgraph.trainer as trainer


class Config:
    data_dir: Path

    def ml_100k_dir(self) -> Path:
        return Path(self.data_dir, "ml-100k")


def main():
    logging.getLogger().setLevel(logging.INFO)
    
    cfg = _config()
    parser = graph.MovieLensParser(cfg.ml_100k_dir())

    dataset = parser.parse()
    training_loader = torch_data.DataLoader(dataset, batch_size=64, shuffle=True)

    model = transh.TranshModel(dataset.entities_len(), dataset.rel_len(), k=50)
    # TODO: Tweak params using grid search to find hyperparameters.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    
    transh_trainer = transh.Trainer(training_loader, dataset, optimizer, model, margin=1)    
    qpcs = selector.QPCSelector(dataset, model)

    ddgraph_trainer = trainer.DDGraphTrainer(dataset, transh_trainer, qpcs)
    ddgraph_trainer.train(epochs=3)


def _config() -> Config:
    cfg = Config()
    cfg.data_dir = environ.get("DATA_DIR", "")
    
    return cfg


if __name__ == "__main__":
    main()
