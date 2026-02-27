"""
Export trained GRU model for embedded deployment (Arduino).

Exports model weights as C header arrays that can be included
directly in an Arduino sketch for real-time inference.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODELS_DIR, GRU_HIDDEN_SIZE, GRU_DENSE_SIZE, GRU_OUTPUT_SIZE, SEQ_LEN
from src.models.gru_model import GRUForcePredictor


def export_weights_to_c_header(model_path, output_path, dataset_name="ninapro"):
    """
    Export GRU model weights to a C header file for Arduino.

    Parameters
    ----------
    model_path : str or Path
        Path to saved .pt model checkpoint.
    output_path : str or Path
        Path for output .h file.
    dataset_name : str
        Dataset identifier for comments.
    """
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    state = checkpoint["model_state_dict"]
    input_size = checkpoint.get("input_size", 2)

    model = GRUForcePredictor(
        input_size=input_size,
        hidden_size=GRU_HIDDEN_SIZE,
        dense_size=GRU_DENSE_SIZE,
        output_size=GRU_OUTPUT_SIZE,
    )
    model.load_state_dict(state)
    model.eval()

    lines = []
    lines.append("// Auto-generated model weights for Arduino deployment")
    lines.append(f"// Dataset: {dataset_name}")
    lines.append(f"// Architecture: GRU({input_size}->{GRU_HIDDEN_SIZE},relu)")
    lines.append(f"//              -> Dense({GRU_DENSE_SIZE},relu) -> Dense({GRU_OUTPUT_SIZE})")
    lines.append(f"// Sequence length: {SEQ_LEN}")
    lines.append("")
    lines.append("#ifndef MODEL_WEIGHTS_H")
    lines.append("#define MODEL_WEIGHTS_H")
    lines.append("")
    lines.append(f"#define INPUT_SIZE {input_size}")
    lines.append(f"#define HIDDEN_SIZE {GRU_HIDDEN_SIZE}")
    lines.append(f"#define DENSE_SIZE {GRU_DENSE_SIZE}")
    lines.append(f"#define OUTPUT_SIZE {GRU_OUTPUT_SIZE}")
    lines.append(f"#define SEQ_LEN {SEQ_LEN}")
    lines.append("")

    # Export each parameter tensor
    for name, param in model.named_parameters():
        arr = param.detach().numpy().flatten()
        c_name = name.replace(".", "_")
        shape_str = "x".join(str(s) for s in param.shape)

        lines.append(f"// {name}: shape ({shape_str}), {arr.size} values")
        lines.append(f"const float {c_name}[{arr.size}] = {{")

        # Write values in rows of 8
        for i in range(0, len(arr), 8):
            chunk = arr[i:i+8]
            vals = ", ".join(f"{v:.8f}f" for v in chunk)
            comma = "," if i + 8 < len(arr) else ""
            lines.append(f"    {vals}{comma}")

        lines.append("};")
        lines.append("")

    lines.append("#endif // MODEL_WEIGHTS_H")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Exported {total_params} parameters to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")


def export_scaler_to_header(scaler_path, output_path, name="input"):
    """Export MinMaxScaler parameters to C header."""
    import joblib
    scaler = joblib.load(scaler_path)

    lines = []
    lines.append(f"// MinMaxScaler parameters ({name})")
    lines.append(f"#define {name.upper()}_SCALER_N_FEATURES {scaler.n_features_in_}")

    scale = scaler.scale_
    min_vals = scaler.min_

    lines.append(f"const float {name}_scale[{len(scale)}] = {{")
    lines.append("    " + ", ".join(f"{v:.8f}f" for v in scale))
    lines.append("};")

    lines.append(f"const float {name}_min[{len(min_vals)}] = {{")
    lines.append("    " + ", ".join(f"{v:.8f}f" for v in min_vals))
    lines.append("};")

    with open(output_path, "a") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Appended {name} scaler to {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export model weights for Arduino")
    parser.add_argument("--dataset", default="ninapro", choices=["ninapro", "hyser", "ghorbani"])
    parser.add_argument("--output", default=None, help="Output .h file path")
    args = parser.parse_args()

    model_path = MODELS_DIR / f"gru_{args.dataset}_best.pt"
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Train the model first: python -m src.models.train --dataset " + args.dataset)
        return

    output_path = args.output or str(
        PROJECT_ROOT / "hardware" / "arduino" / "model_weights.h"
    )

    export_weights_to_c_header(model_path, output_path, args.dataset)


if __name__ == "__main__":
    main()
