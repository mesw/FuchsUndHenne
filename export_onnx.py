"""ONNX Exporter & Numerical Validator for Fuchs und Henne Actor Models.

Strips the Critic head and exports the Actor head with embedded logit masking
and Softmax to compact ONNX graphs (fox_actor.onnx and hens_actor.onnx).
Validates numerical fidelity against ONNX Runtime.
"""

import argparse
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import torch

from fuchs_und_henne.board import NUM_ACTIONS
from fuchs_und_henne.models import ActorCritic, ExportableActor


def export_and_validate(
    weights_path: Path,
    output_onnx_path: Path,
    role_name: str,
) -> None:
    print(f"\n=======================================================")
    print(f"[*] Exporting {role_name.upper()} Model from: {weights_path}")
    print(f"=======================================================")

    # 1. Instantiate and load full ActorCritic weights
    full_model = ActorCritic()
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    full_model.load_state_dict(state_dict)
    full_model.eval()

    # 2. Extract stripped ExportableActor (Conv2D + Actor head + embedded masking + Softmax)
    actor_export = ExportableActor(full_model)
    actor_export.eval()

    # 3. Create dummy inputs matching the specification
    # obs: Float32 (1, 4, 7, 7)
    # mask: Float32 (1, 528)
    dummy_obs = torch.zeros((1, 4, 7, 7), dtype=torch.float32)
    # Set a few sample legal actions in the mask
    dummy_mask = torch.zeros((1, NUM_ACTIONS), dtype=torch.float32)
    dummy_mask[0, 10] = 1.0
    dummy_mask[0, 50] = 1.0
    dummy_mask[0, 120] = 1.0

    # 4. Export to ONNX
    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        actor_export,
        (dummy_obs, dummy_mask),
        str(output_onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["observation", "action_mask"],
        output_names=["action_probabilities"],
        dynamic_axes=None,  # Fixed shape (1, ...) optimized for WASM
    )

    file_size_kb = output_onnx_path.stat().st_size / 1024.0
    print(f"[+] Successfully exported ONNX graph to: {output_onnx_path} ({file_size_kb:.1f} KB)")

    # 5. Check ONNX graph validity
    onnx_model = onnx.load(str(output_onnx_path))
    onnx.checker.check_model(onnx_model)
    print("[+] ONNX model checker passed: graph is structurally valid.")

    # 6. Numerical validation against PyTorch forward pass
    with torch.no_grad():
        pytorch_output = actor_export(dummy_obs, dummy_mask).numpy()

    ort_session = ort.InferenceSession(str(output_onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {
        "observation": dummy_obs.numpy(),
        "action_mask": dummy_mask.numpy(),
    }
    ort_output = ort_session.run(None, ort_inputs)[0]

    max_diff = np.max(np.abs(pytorch_output - ort_output))
    print(f"[+] Numerical difference (PyTorch vs ONNX Runtime): {max_diff:.2e}")
    assert max_diff < 1e-5, f"Discrepancy too large ({max_diff})!"

    # Verify illegal actions are strictly 0.0 probability
    illegal_indices = np.where(dummy_mask[0].numpy() == 0.0)[0]
    assert np.allclose(ort_output[0, illegal_indices], 0.0, atol=1e-6), "Illegal actions had non-zero probability!"
    print(f"[+] Invalid action masking verified: all illegal actions have exactly 0.0 probability.")
    # 7. Also export raw Float32 binary file for zero-dependency C++/WASM inference
    bin_path = output_onnx_path.with_suffix(".bin")
    with open(bin_path, "wb") as f:
        for name, param in actor_export.state_dict().items():
            f.write(param.cpu().float().numpy().tobytes())
    print(f"[+] Exported raw binary weights for C++ to: {bin_path} ({bin_path.stat().st_size / 1024:.1f} KB)")

    # 8. Auto-synchronize raw binary weights to Qt project & build directories
    repo_root = output_onnx_path.resolve().parent.parent if output_onnx_path.resolve().parent.name == "exported_models" else output_onnx_path.resolve().parent
    import shutil
    targets = [repo_root / "QtFuchsUndHenne" / bin_path.name]
    build_dir = repo_root / "QtFuchsUndHenne" / "build"
    if build_dir.exists():
        for d in build_dir.iterdir():
            if d.is_dir() and ("Desktop_" in d.name or "WebAssembly_" in d.name):
                targets.append(d / bin_path.name)

    for t in targets:
        try:
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(bin_path, t)
            print(f"[+] Auto-synchronized weights to: {t}")
        except Exception:
            pass

    print(f"[+] {role_name.upper()} export verified successfully!")


def main():
    parser = argparse.ArgumentParser(description="Export trained PyTorch actors to ONNX graphs")
    parser.add_argument("--fox_weights", "--fox", type=str, default="fox_final.pt", help="Path to Fox weights")
    parser.add_argument("--hen_weights", "--hen", "--hens", type=str, default="hens_final.pt", help="Path to Hen weights")
    parser.add_argument("--output_dir", type=str, default="exported_models", help="Output directory for ONNX models")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    fox_pt = Path(args.fox_weights)
    hen_pt = Path(args.hen_weights)

    if not fox_pt.exists() or not hen_pt.exists():
        print(f"[!] Warning: {fox_pt} or {hen_pt} not found. Checking checkpoints/ for latest files...")
        from spectator import find_latest_checkpoint
        fox_pt = find_latest_checkpoint("foxes") or fox_pt
        hen_pt = find_latest_checkpoint("hens") or hen_pt

    export_and_validate(fox_pt, out_dir / "fox_actor.onnx", "Fox")
    export_and_validate(hen_pt, out_dir / "hens_actor.onnx", "Hens")
    print("\n[*] All models successfully exported to ONNX format!")


if __name__ == "__main__":
    main()
