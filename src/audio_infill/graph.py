import torch
from torchviz import make_dot
from pathlib import Path

from audio_infill.train import JointCodebookInfiller

K = 8
bins = 1024
B = 2
T = 128

model = JointCodebookInfiller(
    K=K,
    bins=bins,
    d_model=512,
    n_heads=8,
    n_layers=8,
    max_len=2048,
    dropout=0.1,
)

dummy_input = torch.randint(0, bins, (B, K, T))
output = model(dummy_input)

out_dir = Path("docs/figures")
out_dir.mkdir(parents=True, exist_ok=True)
dot_path = out_dir / "network.dot"
png_base = out_dir / "network"

dot = make_dot(output, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
dot.save(str(dot_path))
print(f"Saved: {dot_path}")
try:
    dot.render(str(png_base), format="png")
    print(f"Saved: {png_base}.png")
except Exception as e:
    print(f"Could not render PNG (install graphviz system package): {e}")
    print(f"You can render manually: dot -Tpng {dot_path} -o {png_base}.png")