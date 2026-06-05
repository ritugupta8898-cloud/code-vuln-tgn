from src.graph_builder import get_loaders

train_loader, val_loader, test_loader = get_loaders()

# check first batch
for batch in train_loader:
    print(batch)
    print(f'x shape: {batch.x.shape}')
    print(f'edge_index shape: {batch.edge_index.shape}')
    print(f'y shape: {batch.y.shape}')
    break