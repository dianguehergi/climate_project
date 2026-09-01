#!/usr/bin/env bash

DRIVER_VERSION="580.126.09"
DRIVER_ROOT="$HOME/climate_project/archive_old/tcn_convlstm/vendor/NVIDIA-Linux-x86_64-${DRIVER_VERSION}"

if [ ! -d "$DRIVER_ROOT" ]; then
    echo "ERREUR : dossier du pilote introuvable :"
    echo "$DRIVER_ROOT"
    return 1 2>/dev/null || exit 1
fi

required_files=(
    "$DRIVER_ROOT/libnvidia-ml.so.${DRIVER_VERSION}"
    "$DRIVER_ROOT/libcuda.so.${DRIVER_VERSION}"
    "$DRIVER_ROOT/nvidia-smi"
)

for required_file in "${required_files[@]}"; do
    if [ ! -f "$required_file" ]; then
        echo "ERREUR : fichier introuvable : $required_file"
        return 1 2>/dev/null || exit 1
    fi
done

# Création des noms attendus par Linux et PyTorch
ln -sfn \
    "libnvidia-ml.so.${DRIVER_VERSION}" \
    "$DRIVER_ROOT/libnvidia-ml.so.1"

ln -sfn \
    "libnvidia-ml.so.${DRIVER_VERSION}" \
    "$DRIVER_ROOT/libnvidia-ml.so"

ln -sfn \
    "libcuda.so.${DRIVER_VERSION}" \
    "$DRIVER_ROOT/libcuda.so.1"

ln -sfn \
    "libcuda.so.${DRIVER_VERSION}" \
    "$DRIVER_ROOT/libcuda.so"

# Évite de dupliquer le chemin lors de plusieurs activations
clean_library_path="$(
    printf '%s' "${LD_LIBRARY_PATH:-}" |
    tr ':' '\n' |
    grep -vF "$DRIVER_ROOT" |
    grep -vE '/cuda[^/]*/compat/?$' |
    awk 'NF' |
    paste -sd ':' -
)"

export LD_LIBRARY_PATH="$DRIVER_ROOT"

if [ -n "$clean_library_path" ]; then
    export LD_LIBRARY_PATH="$DRIVER_ROOT:$clean_library_path"
fi

case ":$PATH:" in
    *":$DRIVER_ROOT:"*) ;;
    *) export PATH="$DRIVER_ROOT:$PATH" ;;
esac

hash -r

echo "GPU NVIDIA local activé."
echo "Pilote utilisateur : $DRIVER_VERSION"
echo "Dossier             : $DRIVER_ROOT"
