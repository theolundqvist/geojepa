


from src.data.tiles_datamodule import TilesDataModule


def run():
    data = TilesDataModule("data/tiles/tiny/tasks/max_speed")
    data.setup()
    loader = data.test_dataloader()
    print(len(loader))


if __name__ == "__main__":
    run()
