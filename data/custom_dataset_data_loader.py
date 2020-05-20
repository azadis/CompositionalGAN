# ========================================================
# Compositional GAN
# Data Loader
# By Samaneh Azadi
# ========================================================

import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataset(opt):
    dataset = None
    if opt.dataset_mode == 'comp_decomp_unaligned':
        from data.compose_dataset import ComposeDataset
        dataset = ComposeDataset()
    elif opt.dataset_mode == 'comp_decomp_aligned':
        from data.compose_dataset import ComposeAlignedDataset
        dataset = ComposeAlignedDataset()

    elif opt.dataset_mode == 'AFN':
        from data.AFN_dataset import AFNDataset
        dataset = AFNDataset()
    elif opt.dataset_mode == 'AFNCompose':
        from data.AFN_compose_dataset import AFNComposeDataset
        dataset = AFNComposeDataset()

    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset_mode = opt.dataset_mode
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)


    def __iter__(self):
        for i, (data) in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size:
                break
            yield data
