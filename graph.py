import torch
from torchviz import make_dot
from train import JointCodebookInfiller

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
dot = make_dot(output, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
dot.save("network.dot")
print("Saved: network.dot")
try:
    dot.render("network", format="png")
    print("Saved: network.png")
except Exception as e:
    print(f"Could not render PNG (install graphviz system package): {e}")
    print("You can render manually: dot -Tpng network.dot -o network.png")  