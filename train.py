import torch
from src.graph_builder import get_loaders
from src.graph_builder import load_samples
from baseline_model import FraudGNN
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
device  ="cpu"
train_loader, val_loader, test_loader = get_loaders(batch_size=64)
weight = torch.tensor([1.0, 1.2], device=device)
criterion = nn.CrossEntropyLoss(weight=weight)
model = FraudGNN()
optimizer = optim.Adam(model.parameters(), lr=0.01)
model.to(device)
model.train()
print(len(train_loader))
def train(loader):
  print(next(model.parameters()).device)
  for epoch in range(50):
    all_preds=[]
    all_true=[]
    total_loss = 0
    for batch in loader:
        batch = batch.to(device) 
        optimizer.zero_grad()
        out,_ = model(batch.x,batch.edge_index,batch.batch)
        loss = criterion(out,batch.y.squeeze())
        loss.backward()
        optimizer.step()
        preds = out.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_true.extend(batch.y.squeeze().cpu().numpy())
        total_loss += loss.item()

    val_f1 = f1_score(all_true, all_preds, average='macro')
    print(f"epoch: {epoch}, loss: {total_loss/len(loader):.4f} f1:{val_f1:.4f}")


train(train_loader)   
model.eval()
all_preds = []
all_true = []
with torch.no_grad():
    for batch in val_loader:
        batch = batch.to(device)
        out, _ = model(batch.x, batch.edge_index, batch.batch)
        preds = out.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_true.extend(batch.y.squeeze().cpu().numpy())

val_f1 = f1_score(all_true, all_preds, average='macro')
print(f'Val F1: {val_f1:.4f}')