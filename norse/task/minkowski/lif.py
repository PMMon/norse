import torch
import torch.jit
from typing import NamedTuple

import MinkowskiEngine as ME

from norse.torch.functional.lif import LIFParameters

from norse.torch import LIFFeedForwardState, LIFCell


class MinkowskiLIFState(NamedTuple):
    v: torch.Tensor
    i: torch.Tensor
    z: torch.Tensor


class MinkowskiLIFCell(ME.MinkowskiModuleBase):
    def __init__(self, shape, v_th, tau_mem, method):
        super().__init__()
        self.v_th = torch.nn.Parameter(torch.as_tensor(v_th))
        self.tau_mem = torch.nn.Parameter(torch.as_tensor(tau_mem))
        self.module = LIFCell(p=LIFParameters(tau_mem_inv=self.tau_mem, v_th=self.v_th, method=method))
        self.union = ME.MinkowskiUnion()
        self.shape = shape

    def set_batch_size(self, batch_size):
        shape = torch.Size([batch_size, 1] + list(self.shape))
        self.dense = ME.MinkowskiToDenseTensor(shape=shape)

    def forward(self, input, state):
        output, s_out = self.module(self.dense(input), state)
        if torch.sum(output) == 0:
            return ME.to_sparse_all(output), s_out
        else:
            return ME.to_sparse(output), s_out

"""
    def forward(self, input, state):
        if state is not None:
            lifstate = LIFFeedForwardState(state.v.F, state.i.F)

            output, s_out = self.module(input.F, lifstate)

            new_state = MinkowskiLIFState(
                v=ME.SparseTensor(
                    features=s_out.v,
                    coordinates=state.v.C,
                    coordinate_manager=input.coordinate_manager,
                ),
                i=ME.SparseTensor(
                    features=s_out.i,
                    coordinates=state.i.C,
                    coordinate_manager=input.coordinate_manager,
                ),
                z=output,
            )
        else:
            zero_input_tensor = ME.SparseTensor(
                features=torch.zeros_like(input.F),
                coordinates=input.C,
                coordinate_manager=input.coordinate_manager,
            )
            v_tensor, i_tensor = zero_input_tensor, zero_input_tensor
            lifstate = LIFFeedForwardState(v_tensor.F, i_tensor.F)

            output, s_out = self.module(input.F, lifstate)

            new_state = MinkowskiLIFState(
                v=ME.SparseTensor(
                    features=s_out.v,
                    coordinates=input.C,
                    coordinate_manager=input.coordinate_manager,
                ),
                i=ME.SparseTensor(
                    features=s_out.i,
                    coordinates=input.C,
                    coordinate_manager=input.coordinate_manager,
                ),
                z=output,
            )

        return (
            ME.SparseTensor(
                features=output,
                coordinates=input.C,
                coordinate_manager=input.coordinate_manager,
            ),
            new_state
        )
"""