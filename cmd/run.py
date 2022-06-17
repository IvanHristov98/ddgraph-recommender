from os import environ
from pathlib import Path

import ddgraph.graph as graph


class Config:
    data_dir: Path

    def ml_100k_dir(self) -> Path:
        return Path(self.data_dir, "ml-100k")


def main():
    cfg = _config()
    parser = graph.MovieLensParser(cfg.ml_100k_dir())

    g = parser.parse()


def _config() -> Config:
    cfg = Config()
    cfg.data_dir = environ.get("DATA_DIR", "")
    
    return cfg


if __name__ == "__main__":
    main()
