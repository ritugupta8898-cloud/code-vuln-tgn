from tree_sitter import Language, Parser
import tree_sitter_c
import torch
import torch.nn.functional as F
import json
import os
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# constants
NODE_TYPES = [
    'if_statement', 'call_expression', 'assignment_expression',
    'return_statement', 'switch_statement', 'declaration',
    'goto_statement', 'while_statement', 'for_statement',
    'binary_expression', 'unary_expression'
]

DANGEROUS_CALLS = {
    'memcpy', 'memset', 'strcpy', 'strcat', 'sprintf', 'gets',
    'scanf', 'malloc', 'free', 'realloc', 'strlen', 'strncpy'
}

def parse_function(code: str):
    parser = Parser(Language(tree_sitter_c.language()))
    tree = parser.parse(bytes(code, 'utf8'))
    return tree

def extract_nodes(root_node):
    IMPORTANT_TYPES = set(NODE_TYPES)
    nodes = []
    node_to_idx = {}

    def traverse(node):
        if node.type in IMPORTANT_TYPES:
            idx = len(nodes)
            nodes.append({
                'idx': idx,
                'type': node.type,
                'text': node.text.decode('utf8') if node.text else '',
                'start': node.start_point,
                'end': node.end_point
            })
            node_to_idx[node.id] = idx
        for child in node.children:
            traverse(child)

    traverse(root_node)
    return nodes, node_to_idx

def extract_edges(root_node, node_to_idx):
    edges = []

    def traverse(node, parent_idx=None):
        current_idx = node_to_idx.get(node.id)
        if parent_idx is not None and current_idx is not None:
            edges.append((parent_idx, current_idx, 0))
        idx_to_pass = current_idx if current_idx is not None else parent_idx
        prev_sibling_idx = None
        for child in node.children:
            child_idx = node_to_idx.get(child.id)
            if child_idx is not None:
                if prev_sibling_idx is not None:
                    edges.append((prev_sibling_idx, child_idx, 1))
                prev_sibling_idx = child_idx
            traverse(child, idx_to_pass)

    traverse(root_node)
    return edges

def extract_node_features(nodes):
    features = []
    for node in nodes:
        type_vec = [1 if node['type'] == t else 0 for t in NODE_TYPES]
        is_dangerous = 0
        if node['type'] == 'call_expression':
            for kw in DANGEROUS_CALLS:
                if kw in node['text']:
                    is_dangerous = 1
                    break
        features.append(type_vec + [is_dangerous])
    return features

def load_samples(path):
    samples = []
    with open(path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples

def build_dataset(samples):
    dataset = []
    for i, sample in enumerate(samples):
        if i % 1000 == 0:
            print(f'  Processing {i}/{len(samples)}...')
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
        return torch.load(path, weights_only=False)
    return None

def get_loaders(batch_size=32):
    os.makedirs('data/processed', exist_ok=True)

    splits = {
        'train': 'data/raw/train.jsonl',
        'val': 'data/raw/val.jsonl',
        'test': 'data/raw/test.jsonl'
    }

    datasets = {}
    for split, path in splits.items():
        cache_path = f'data/processed/{split}.pt'
        cached = load_dataset_cached(cache_path)
        if cached is None:
            samples = load_samples(path)
            print(f'Building {split} dataset ({len(samples)} samples)...')
            dataset = build_dataset(samples)
            save_dataset(dataset, cache_path)
        else:
            dataset = cached
        datasets[split] = dataset

    train_loader = DataLoader(datasets['train'], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(datasets['val'], batch_size=batch_size)
    test_loader = DataLoader(datasets['test'], batch_size=batch_size)

    return train_loader, val_loader, test_loader