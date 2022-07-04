import logging
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
    logging.getLogger().setLevel(logging.INFO)
    
    cfg = _config()
    parser = graph.MovieLensParser(cfg.ml_100k_dir())

    onto = parser.parse()
    dataset = graph.TripletDataset(onto)
    training_loader = torch_data.DataLoader(dataset, batch_size=64, shuffle=True)

    model = transe.TranseModel(onto.entities_len(), onto.relations_len(), k=50)
    # TODO: Tweak params using grid search to find hyperparameters.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    
    trainer = transe.Trainer(training_loader, onto, optimizer, model, margin=1)
    calc = transe.Calculator(onto, sample_size=64)

    metrics_bundle = calc.calculate(trainer.model())
    logging.info(f"Hits@10 ---> {metrics_bundle.hits_at_10}")
    logging.info(f"Rank ---> {metrics_bundle.mean_rank}")
    
    for i in range(10):
        trainer.train_one_epoch()

    metrics_bundle = calc.calculate(trainer.model())
    logging.info(f"Hits@10 ---> {metrics_bundle.hits_at_10}")
    logging.info(f"Rank ---> {metrics_bundle.mean_rank}")


def _config() -> Config:
    cfg = Config()
    cfg.data_dir = environ.get("DATA_DIR", "")
    
    return cfg


if __name__ == "__main__":
    main()
