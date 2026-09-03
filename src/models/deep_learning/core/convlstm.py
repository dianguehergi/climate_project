from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class ConvLSTMCell(nn.Module):
    """
    Cellule ConvLSTM.

    Les opérations linéaires classiques d'un LSTM sont
    remplacées par des convolutions 2D afin de préserver
    la structure spatiale du patch.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError(
                "kernel_size doit être impair afin de "
                "conserver les dimensions spatiales."
            )

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        padding = kernel_size // 2

        self.gate_convolution = nn.Conv2d(
            in_channels=(
                input_channels
                + hidden_channels
            ),
            out_channels=(
                4 * hidden_channels
            ),
            kernel_size=kernel_size,
            padding=padding,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(
            self.gate_convolution.weight
        )

        if self.gate_convolution.bias is not None:
            nn.init.zeros_(
                self.gate_convolution.bias
            )

    def initialize_state(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = inputs.shape[0]
        height = inputs.shape[2]
        width = inputs.shape[3]

        state_shape = (
            batch_size,
            self.hidden_channels,
            height,
            width,
        )

        hidden_state = inputs.new_zeros(
            state_shape
        )

        cell_state = inputs.new_zeros(
            state_shape
        )

        return hidden_state, cell_state

    def forward(
        self,
        inputs: torch.Tensor,
        state: tuple[
            torch.Tensor,
            torch.Tensor,
        ] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 4:
            raise ValueError(
                "Une cellule ConvLSTM attend une entrée "
                "(batch, canaux, hauteur, largeur). "
                f"Forme reçue : {inputs.shape}"
            )

        if state is None:
            hidden_state, cell_state = (
                self.initialize_state(inputs)
            )
        else:
            hidden_state, cell_state = state

        combined = torch.cat(
            [
                inputs,
                hidden_state,
            ],
            dim=1,
        )

        gates = self.gate_convolution(
            combined
        )

        (
            input_gate,
            forget_gate,
            output_gate,
            candidate_gate,
        ) = torch.chunk(
            gates,
            chunks=4,
            dim=1,
        )

        input_gate = torch.sigmoid(
            input_gate
        )

        forget_gate = torch.sigmoid(
            forget_gate
        )

        output_gate = torch.sigmoid(
            output_gate
        )

        candidate_gate = torch.tanh(
            candidate_gate
        )

        new_cell_state = (
            forget_gate * cell_state
            + input_gate * candidate_gate
        )

        new_hidden_state = (
            output_gate
            * torch.tanh(new_cell_state)
        )

        return (
            new_hidden_state,
            new_cell_state,
        )


class ConvLSTMRegressor(nn.Module):
    """
    ConvLSTM empilé pour prévoir la température au centre
    du patch le jour suivant.

    Entrée :
        batch × temps × canaux × hauteur × largeur

    Sortie :
        une température normalisée par séquence
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: Sequence[int] = (
            32,
            64,
        ),
        kernel_size: int = 3,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()

        if not hidden_channels:
            raise ValueError(
                "hidden_channels ne peut pas être vide."
            )

        cells = []

        current_channels = input_channels

        for output_channels in hidden_channels:
            cells.append(
                ConvLSTMCell(
                    input_channels=current_channels,
                    hidden_channels=output_channels,
                    kernel_size=kernel_size,
                )
            )

            current_channels = output_channels

        self.cells = nn.ModuleList(
            cells
        )

        self.spatial_dropout = nn.Dropout2d(
            dropout
        )

        self.regression_head = nn.Sequential(
            nn.Linear(
                current_channels,
                32,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                32,
                1,
            ),
        )

        self.hidden_channels = tuple(
            hidden_channels
        )

        self.kernel_size = kernel_size

        self.reset_head_parameters()

    def reset_head_parameters(self) -> None:
        for module in self.regression_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(
                    module.weight
                )

                if module.bias is not None:
                    nn.init.zeros_(
                        module.bias
                    )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        if inputs.ndim != 5:
            raise ValueError(
                "Le ConvLSTM attend une entrée "
                "(batch, temps, canaux, hauteur, largeur). "
                f"Forme reçue : {inputs.shape}"
            )

        _, sequence_length, _, height, width = (
            inputs.shape
        )

        states: list[
            tuple[
                torch.Tensor,
                torch.Tensor,
            ]
            | None
        ] = [
            None
            for _ in self.cells
        ]

        for time_index in range(
            sequence_length
        ):
            layer_inputs = inputs[
                :,
                time_index,
                :,
                :,
                :,
            ]

            for layer_index, cell in enumerate(
                self.cells
            ):
                hidden_state, cell_state = cell(
                    layer_inputs,
                    states[layer_index],
                )

                states[layer_index] = (
                    hidden_state,
                    cell_state,
                )

                layer_inputs = hidden_state

                if layer_index < len(
                    self.cells
                ) - 1:
                    layer_inputs = (
                        self.spatial_dropout(
                            layer_inputs
                        )
                    )

        final_hidden_state = states[-1][0]

        center_row = height // 2
        center_column = width // 2

        center_features = final_hidden_state[
            :,
            :,
            center_row,
            center_column,
        ]

        predictions = self.regression_head(
            center_features
        )

        return predictions.squeeze(-1)
