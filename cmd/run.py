from os import environ
from pathlib import Path

import ddgraph.graph.parser as parse


class Config:
    data_dir: Path

    def ml_100k_dir(self) -> Path:
        return Path(self.data_dir, "ml-100k")


def main():
    cfg = _config()
    parser = parse.MovieLensParser(cfg.ml_100k_dir())

    graph = parser.parse()


def _config() -> Config:
    cfg = Config()
    cfg.data_dir = environ.get("DATA_DIR", "")
    
    return cfg


if __name__ == "__main__":
    main()
