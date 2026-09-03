#!/usr/bin/env bash

# Le serveur utilise maintenant le pilote NVIDIA système.
# On supprime les anciennes bibliothèques locales qui causaient
# un conflit Driver/library version mismatch.

unset LD_LIBRARY_PATH

echo "GPU NVIDIA système activé."
nvidia-smi | head -n 12
