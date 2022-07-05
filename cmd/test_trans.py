import logging
from os import environ
from pathlib import Path
import torch

import torch.utils.data as torch_data

import ddgraph.graph as graph
import ddgraph.transh as transh


class Config:
    data_dir: Path
    onto_name: str

    def ml_100k_dir(self) -> Path:
        return Path(self.data_dir, "ml-100k")

    def wn_dir(self) -> Path:
        return Path(self.data_dir, "wordnet")


def main():
    logging.getLogger().setLevel(logging.INFO)
    
    cfg = _config()

    onto = _onto(cfg)

    dataset = graph.TripletDataset(onto)
    training_loader = torch_data.DataLoader(dataset, batch_size=64, shuffle=True)

    model = transh.TranshModel(onto.entities_len(), onto.relations_len(), k=50)
    # TODO: Tweak params using grid search to find hyperparameters.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    
    trainer = transh.Trainer(training_loader, onto, optimizer, model, margin=1)
    calc = transh.Calculator(onto, sample_size=512)

    # metrics_bundle = calc.calculate(trainer.model())
    # logging.info(f"Hits@10 ---> {metrics_bundle.hits_at_10}")
    # logging.info(f"Rank ---> {metrics_bundle.mean_rank}")
    
    for i in range(20):
        trainer.train_one_epoch()

    metrics_bundle = calc.calculate(trainer.model())
    logging.info(f"Hits@10 ---> {metrics_bundle.hits_at_10}")
    logging.info(f"Rank ---> {metrics_bundle.mean_rank}")


def _config() -> Config:
    cfg = Config()
    cfg.data_dir = environ.get("DATA_DIR", "")
    cfg.onto_name = environ.get("ONTO_NAME", "movielens")
    
    return cfg


def _onto(cfg: Config) -> graph.Ontology:
    if cfg.onto_name == "movielens":
        parser = graph.MovieLensParser(cfg.ml_100k_dir())
        return parser.parse()

    if cfg.onto_name == "wordnet":
        parser = graph.WNParser(cfg.wn_dir())
        return parser.parse()
    
    raise Exception(f"unsupported ontology {cfg.onto_name}")


if __name__ == "__main__":
    main()
