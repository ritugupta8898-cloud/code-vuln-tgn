import os
import torch
import json
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def load_samples(path):
    samples = []
    with open(path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def build_dataset(samples):
    dataset = []

    for sample in samples:
        tree = parse_function(sample["func"])
        nodes, node_to_idx = extract_nodes(tree.root_node)
        edges = extract_edges(tree.root_node, node_to_idx)
        x = extract_node_features(nodes)

        if len(nodes) == 0:
            continue

        if len(edges) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 2), dtype=torch.float)
        else:
            edge_index = torch.tensor(
                [[src for src, dst, t in edges],
                 [dst for src, dst, t in edges]],
                dtype=torch.long
            )
            edge_types = torch.tensor([t for src, dst, t in edges], dtype=torch.long)
            edge_attr = F.one_hot(edge_types, num_classes=2).float()

        graph = Data(
            x=torch.tensor(x, dtype=torch.float),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([int(sample["target"])], dtype=torch.long)
        )

        dataset.append(graph)

    return dataset

def save_dataset(dataset, path):
    torch.save(dataset, path)
    print(f'Saved {len(dataset)} graphs to {path}')

def load_dataset_cached(path):
    if os.path.exists(path):
        print(f'Loading cached dataset from {path}')
        return torch.load(path)
    return None


def get_loaders(batch_size=32):
    os.makedirs('data/processed', exist_ok=True)
    
    for split, path in [('train', 'data/raw/train.jsonl'), 
                         ('val', 'data/raw/val.jsonl'),
                         ('test', 'data/raw/test.jsonl')]:
        cache_path = f'data/processed/{split}.pt'
        cached = load_dataset_cached(cache_path)
        
        if cached is None:
            samples = load_samples(path)
            print(f'Building {split} dataset...')
            dataset = build_dataset(samples)
            save_dataset(dataset, cache_path)
        else:
            dataset = cached
        
        if split == 'train':
            train_dataset = dataset
        elif split == 'val':
            val_dataset = dataset
        else:
            test_dataset = dataset
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader
