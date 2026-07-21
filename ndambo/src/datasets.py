from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class NumpySequenceDataset(Dataset):
    """
    Dataset PyTorch construit à partir de séquences NumPy.

    X attendu :
        (nombre_sequences, nombre_variables, longueur_sequence)

    y attendu :
        (nombre_sequences,)
    """

    def __init__(
        self,
        x_array: np.ndarray,
        y_array: np.ndarray,
    ) -> None:
        if x_array.ndim != 3:
            raise ValueError(
                "X doit avoir la forme "
                "(séquences, variables, longueur). "
                f"Forme reçue : {x_array.shape}"
            )

        if y_array.ndim != 1:
            raise ValueError(
                "y doit avoir une seule dimension. "
                f"Forme reçue : {y_array.shape}"
            )

        if len(x_array) != len(y_array):
            raise ValueError(
                "X et y doivent contenir le même nombre "
                "d'observations."
            )

        self.x_tensor = torch.from_numpy(
            x_array.astype(np.float32, copy=False)
        )

        self.y_tensor = torch.from_numpy(
            y_array.astype(np.float32, copy=False)
        )

    def __len__(self) -> int:
        return len(self.y_tensor)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.x_tensor[index],
            self.y_tensor[index],
        )