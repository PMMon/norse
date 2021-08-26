"""
An example of using sparse event-based Dynamic Vision System (DVS) data with
NVIDIA's MinkowskiEngine to implement sparse convolutions in a spiking neural
network.

See https://github.com/NVIDIA/MinkowskiEngine
"""

from argparse import ArgumentParser

import torch
import numpy as np
import MinkowskiEngine as ME
import pytorch_lightning as pl

# Tonic is a great library for event-based datasets
#   https://github.com/neuromorphs/tonic
import tonic
import tonic.transforms as tonic_transforms

import norse.torch as norse
from norse.torch.functional.lif import LIFParameters
from norse.torch.module.leaky_integrator import LILinearCell

from minkowski.lif import MinkowskiLIFCell


class SNNMinkowski(ME.MinkowskiNetwork):
    def __init__(self, in_channels, out_channels=128, out_features=10, D=2, method='super', alpha=100., v_th = 1., tau_mem = 100., width=34, height=34):
        super().__init__(D=D)
        self.conv1 = ME.MinkowskiConvolution(in_channels, 32, 5, dimension=D)
        self.conv2 = ME.MinkowskiConvolution(32, 64, 5, stride=2, dimension=D)
        self.conv3 = ME.MinkowskiConvolution(64, out_channels, kernel_size=3, dimension=D)

        self.batchnorm1 = ME.MinkowskiBatchNorm(64)
        self.batchnorm2 = ME.MinkowskiBatchNorm(out_channels)

        self.fc1 = ME.MinkowskiLinear(128, out_features)

        self.lrelu = ME.MinkowskiLeakyReLU()

        self.lif1 = MinkowskiLIFCell((width, height), v_th, tau_mem, method)
        self.lif2 = MinkowskiLIFCell((width, height), v_th, tau_mem, method)
        #self.lif3 = MinkowskiLIFCell((width, height), v_th, tau_mem, method)

        self.loc_pool = ME.MinkowskiAvgPooling([2, 2], stride=1, dimension=D)
        self.glob_pool = ME.MinkowskiMaxPooling([width, height], stride=[width, height], dimension=D) #ME.MinkowskiGlobalPooling()

        self.to_feature = ME.MinkowskiToFeature()

        self.out = LILinearCell(out_features, out_features)

        self.out_features = out_features

        self.union = ME.MinkowskiUnion()



    def forward(self, x):
        batch_size, seq_len, channel, _, _ = x.shape

        # initial state
        s1, s2, s3, sout = None, None, None, None

        # Force TBCXY format
        dense = x.to_dense().permute(1, 0, 2, 3, 4)

        # times = set(sparse_input.C[:, 0].cpu().tolist())

        self.lif1.set_batch_size(batch_size)
        self.lif2.set_batch_size(batch_size)
        #self.lif3.set_batch_size(batch_size)

        voltage = torch.zeros(seq_len, batch_size, self.out_features)

        for t_step in range(dense.shape[0]): # in times:
            sparse_in = self.pad_batch(None, dense[t_step], batch_size)
            #print("sparse_in: " + str(sparse_in))
            #print("dim: " + str(sparse_in.shape))

            z = self.batchnorm1(self.conv2(self.conv1(sparse_in)))
            #print(z.shape)

            #print("process first LIF")
            z, s1 = self.lif1(z, s1)
            z = self.pad_batch(z, dense[t_step], batch_size)
            #print("z: " + str(z))

            #print(z.shape)
            #print("process local pool")
            z = self.loc_pool(z)
            #print("z: " + str(z))
            #print(z.shape)

            z = self.batchnorm2(self.conv3(z))
            #print(z.shape)

            #print("process second LIF")
            z, s2 = self.lif2(z, s2)
            #print("z: " + str(z))
            z = self.pad_batch(z, dense[t_step], batch_size)
            #print("z: " + str(z))
            #print(z.shape)
            z = self.glob_pool(z)
            #print(z.shape)

            #print("process fc")
            z = self.fc1(z)
            z = self.lrelu(z)

            #print("process third LIF")
            #z, s3 = self.lif3(z, s3)
            #z = self.pad_batch(z, dense[t_step], batch_size)

            z = self.to_feature(z)
            #print(z.shape)

            v_out, sout = self.out(z, sout)

            #print("final output: " + str(v_out))

            #print("output: " + str(v_out))
            #print("dim: " + str(v_out.shape))
            #print(" ============ " + str(t_step) + " ==============")

            voltage[t_step, :, :] = v_out

        #print("v_out: " + str(v_out))

        return voltage


    def pad_batch(self, sparse_in, dense_in, batch_size):
        if torch.sum(dense_in) == 0 and not sparse_in:
            #print("empty input!")
            #print("shape: " + str(dense_in.shape))
            """
            padding = ME.SparseTensor(
                features=torch.zeros(batch_size, 1).type_as(sparse_input.F),
                coordinates=torch.cat((torch.arange(0, batch_size).unsqueeze(1), torch.zeros(batch_size, 2)), 1).type_as(sparse_input.C),#.type_as(sparse_input.C)
                coordinate_manager=sparse_input.coordinate_manager
            )
            print("padding: " + str(padding))
            """
            return self.glob_pool(ME.to_sparse_all(dense_in))

            return self.union(sparse_in, padding)
        else:
            if not sparse_in:
                sparse_in = ME.to_sparse(dense_in)
            batch_nr = sparse_in.C[:, 0].unique()
            if len(batch_nr) != batch_size:
                coords = [i for i in range(0, batch_size) if i not in batch_nr]
                padding = ME.SparseTensor(
                    features=torch.zeros(batch_size - len(batch_nr), 1).type_as(sparse_in.F),
                    coordinates=torch.cat((torch.tensor(coords).unsqueeze(1), torch.zeros(batch_size - len(batch_nr), 2)), 1).type_as(sparse_in.C),
                    coordinate_manager=sparse_in.coordinate_manager
                )

                return self.union(sparse_in, padding)
            else:
                return sparse_in


class NMNISTModule(pl.LightningModule):
    def __init__(
            self,
            model,
            batch_size,
            transform,
            lr=1e-2,
            weight_decay=1e-5,
            data_root="./data",
            train_split=1.0
    ):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.data_root = data_root

        self.criterion = torch.nn.functional.nll_loss
        self.transform = transform

        self.train_split = train_split


    def forward(self, x):
        voltages = self.model(x)
        #m, _ = torch.max(voltages, 0)
        m = voltages[-1]
        log_p_y = torch.nn.functional.log_softmax(m, dim=1)
        return log_p_y

    # def batch_collate_fn(self, list_data):
    #     batches = []
    #     for batch_idx in range(len(list_data[0][0])):
    #         chunks = []
    #         for datapoint in list_data:
    #             if batch_idx < len(datapoint[0]):
    #                 chunks.append(datapoint[0][batch_idx])
    #         len_diff = self.batch_size - len(chunks)
    #         while len_diff > 0:
    #             chunks.append(
    #                 [
    #                     torch.zeros((1, 20), dtype=torch.float32),
    #                     torch.zeros((1, 2), dtype=torch.float32),
    #                 ]
    #             )
    #             len_diff -= 1
    #         batches.append(ME.utils.batch_sparse_collate(chunks))
    #     labels = [x[1] for x in list_data]
    #     return batches, torch.stack(labels).squeeze()

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            nmnist_full = tonic.datasets.NMNIST(save_to=self.data_root, train=True, transform=self.transform)
            self.nmnist_train, self.mnist_val = torch.utils.data.random_split(nmnist_full, [int(self.train_split * len(nmnist_full)), int((1 - self.train_split) * len(nmnist_full))])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.nmnist_train,
            collate_fn=tonic.utils.pad_tensors,
            batch_size=self.batch_size,
            shuffle=True,
        )

    # def val_dataloader(self):
    #     return DataLoader(
    #         tonic.datasets.NMNIST(save_to="./data", train=False, transform=transform),
    #         collate_fn=single_chunk_collate
    #         if self.timesteps <= 1
    #         else batch_collate_fn(self.batch_size),
    #         batch_size=self.batch_size,
    #         shuffle=True,
    #     )

    def training_step(self, batch, batch_idx):
        chunks, labels = batch
        # Must clear cache at regular interval
        if self.global_step % 10 == 0:
            torch.cuda.empty_cache()
        log_p = self(chunks)
        labels = torch.tensor(labels)
        loss = self.criterion(log_p, labels)
        self.log('train_loss', loss)
        return loss

    # def validation_step(self, batch, batch_idx):
    #     chunks, labels = batch
    #     # Must clear cache at regular interval
    #     if self.global_step % 10 == 0:
    #         torch.cuda.empty_cache()
    #     out, _ = self(chunks)
    #     loss = self.criterion(out, labels)
    #     acc = torch.eq(out.detach().argmax(1), labels).float().mean()
    #     self.log("VAcc", acc)
    #     return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
        return [optimizer], [scheduler]


def main(args):
    transform = tonic_transforms.Compose(
        [
            tonic_transforms.Denoise(filter_time=10000),
            tonic_transforms.Subsample(args.subsample),
            tonic_transforms.ToSparseTensor(merge_polarities=True),
        ]
    )

    network = SNNMinkowski(in_channels=1)
    module = NMNISTModule(network, batch_size=args.batch_size, transform=transform, data_root=args.data_root, weight_decay=args.weight_decay)

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(module)


if __name__ == "__main__":
    parser = ArgumentParser("N-MNIST training")
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument(
        "--subsample",
        type=float,
        default=1e-3,
        help="Supsampling multiplier for timesteps. 0.1 = 10\% of timesteps. Defaults to 1e-3",
    )
    parser.add_argument(
        "--model",
        default="snn",
        choices=["snn", "ann"],
        help="Model to use for training. Defaults to snn.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="The root of data for the NMNIST dataset. Defaults to ./data",
    )
    parser.add_argument(
        "--weight_decay",
        default=1e-5,
        type=float,
        help="Weight decay of optimizer",
    )
    parser.add_argument(
        "--lr",
        default=1e-2,
        type=float,
        help="Learning rate",
    )
    parser.add_argument(
        "--train_split",
        default=1.0,
        type=float,
        help="Rate to split train and validation set (specifies train set size)",
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Batch size",
    )
    args = parser.parse_args()
    main(args)