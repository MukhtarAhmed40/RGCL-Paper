import torch
import yaml
from models.rgcl_model import RGCL
from utils.data_utils import load_data, augment_graph

config = yaml.safe_load(open("configs/default.yaml"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = load_data()

model = RGCL(in_dim=data.num_features, hidden_dim=config["hidden_dim"]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

for epoch in range(config["epochs"]):
    model.train()

    g1 = augment_graph(data)
    g2 = augment_graph(data)

    _, loss = model(g1, g2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
