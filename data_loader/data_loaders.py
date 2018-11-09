from base import BaseDataLoader
import dataset
from data_loader.collates import PadCollate
from torchvision import transforms


class PMEmoDataLoader(BaseDataLoader):
    def __init__(self, batch_size, shuffle, validation_split, num_workers, collate_fn=PadCollate(dim=0),
                 load_transformed='Spec',  # avoid on-the-fly transform
                 data_dir='../pmemo_dataset/PMEmo'):
        # Currently assume the audio data is transformed and stored beforehand;
        # this avoids transforming on-the-fly, which slows training.
        self.transform = transforms.Compose([dataset.transformers.LoadTensor()])

        self.data_dir = data_dir
        self.load_transformed = load_transformed
        self.dataset = dataset.datasets.PMEmodata(self.data_dir,
                                                  load_transformed=self.load_transformed,
                                                  transform=self.transform)
        super(PMEmoDataLoader, self).__init__(self.dataset,
                                              batch_size, shuffle,
                                              validation_split, num_workers,
                                              collate_fn
                                              )


if __name__ == '__main__':
    dl = PMEmoDataLoader(batch_size=2, shuffle=False,
                         validation_split=0.1, num_workers=0)
    print(dl.init_kwargs)
    val_dl = dl.split_validation()

    batch = next(iter(dl))
    val_batch = next(iter(val_dl))

    print("\nExample of training sample")
    print(batch[0], batch[1], batch[2], batch[3], batch[4])
    print(batch[0].size(), batch[4].size())

    print("\nExample of validation sample")
    print(val_batch[0], val_batch[1], val_batch[2], val_batch[3], val_batch[4])
    print(val_batch[0].size(), val_batch[4].size())
