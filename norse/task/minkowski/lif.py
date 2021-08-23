import torch
import torch.jit
from typing import NamedTuple

import MinkowskiEngine as ME

from norse.torch import LIFFeedForwardState, LIFCell


class MinkowskiLIFState(NamedTuple):
    v: torch.Tensor
    i: torch.Tensor
    z: torch.Tensor


class MinkowskiLIFCell(ME.MinkowskiModuleBase):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.module = LIFCell(*args, **kwargs)
        self.union = ME.MinkowskiUnion()

    def forward(self, input, state):
        if state is not None:
            """
            # Align state with input
            zero_state_tensor = ME.SparseTensor(
                features=torch.zeros_like(state.v.F),
                coordinates=state.v.C,
                coordinate_manager=input.coordinate_manager,
            )
            print("v: " + str(state.v))
            v_in = ME.SparseTensor(
                    features=state.v.F,
                    coordinates=state.v.C,
                    coordinate_manager=input.coordinate_manager,
                )
            print("v_in: " + str(v_in))
            #print("v coordinates: " + str(state.v.C))
            #print("v features: " + str(state.v.F))
            print("input: " + str(input))
            #print("input coordinates: " + str(input.C))
            #print("input features: " + str(input.F))
            v_tensor = self.union(
                input,
                v_in
            )
            print("v_tensor: " + str(v_tensor))
            i_tensor = self.union(
                input,
                ME.SparseTensor(
                    features=state.i.F,
                    coordinates=state.i.C,
                    coordinate_manager=input.coordinate_manager,
                ),
            )
            state = LIFFeedForwardState(v_tensor.F, i_tensor.F)
            input = self.union(input, zero_state_tensor)
            """
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
