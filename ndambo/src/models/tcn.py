from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CausalConv1d(nn.Module):
    """
    Convolution 1D causale.

    Le padding est ajouté uniquement à gauche afin que
    la sortie à un instant donné ne dépende jamais
    d'une information future.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()

        if kernel_size < 2:
            raise ValueError(
                "kernel_size doit être supérieur ou égal à 2."
            )

        self.left_padding = (
            kernel_size - 1
        ) * dilation

        self.convolution = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        padded_inputs = F.pad(
            inputs,
            pad=(self.left_padding, 0),
        )

        return self.convolution(
            padded_inputs
        )


class TemporalResidualBlock(nn.Module):
    """
    Bloc résiduel contenant deux convolutions causales.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.conv1 = CausalConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

        self.activation1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

        self.activation2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.residual_projection = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )
        else:
            self.residual_projection = nn.Identity()

        self.output_activation = nn.GELU()

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        residual = self.residual_projection(
            inputs
        )

        outputs = self.conv1(
            inputs
        )

        outputs = self.activation1(
            outputs
        )

        outputs = self.dropout1(
            outputs
        )

        outputs = self.conv2(
            outputs
        )

        outputs = self.activation2(
            outputs
        )

        outputs = self.dropout2(
            outputs
        )

        outputs = outputs + residual

        return self.output_activation(
            outputs
        )


class TemporalConvolutionalNetwork(nn.Module):
    """
    TCN pour la prévision d'une valeur continue.

    Entrée :
        batch × variables × longueur

    Sortie :
        une prédiction par séquence
    """

    def __init__(
        self,
        input_channels: int = 1,
        channels: tuple[int, ...] = (32, 64, 64),
        kernel_size: int = 3,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()

        if not channels:
            raise ValueError(
                "La liste des canaux ne peut pas être vide."
            )

        blocks = []

        current_channels = input_channels

        for block_index, output_channels in enumerate(
            channels
        ):
            dilation = 2 ** block_index

            blocks.append(
                TemporalResidualBlock(
                    in_channels=current_channels,
                    out_channels=output_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )

            current_channels = output_channels

        self.temporal_blocks = nn.Sequential(
            *blocks
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

        self.channels = channels
        self.kernel_size = kernel_size

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialise les couches convolutionnelles et linéaires."""

        for module in self.modules():
            if isinstance(
                module,
                (nn.Conv1d, nn.Linear),
            ):
                nn.init.kaiming_normal_(
                    module.weight,
                    nonlinearity="relu",
                )

                if module.bias is not None:
                    nn.init.zeros_(
                        module.bias
                    )

    @property
    def receptive_field(self) -> int:
        """
        Calcule le champ réceptif théorique du réseau.

        Chaque bloc contient deux convolutions.
        """

        dilation_sum = sum(
            2 ** block_index
            for block_index in range(
                len(self.channels)
            )
        )

        return (
            1
            + 2
            * (self.kernel_size - 1)
            * dilation_sum
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError(
                "L'entrée doit avoir la forme "
                "(batch, variables, longueur). "
                f"Forme reçue : {inputs.shape}"
            )

        temporal_features = self.temporal_blocks(
            inputs
        )

        # Dernier instant de la séquence
        last_features = temporal_features[
            :, :, -1
        ]

        predictions = self.regression_head(
            last_features
        )

        return predictions.squeeze(-1)