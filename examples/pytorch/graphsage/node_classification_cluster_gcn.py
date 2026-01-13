
import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.nn.pytorch as dglnn
import time
import argparse
import tqdm
import psutil
import os

from load_graph import load_reddit, load_ogb

class SAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, device):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        The inference code is written in a fashion that it could handle message passing inference for multi-layer GraphSAGE even when
        the input graph is simply a Block, since we support inference for neighbor-sampled subgraphs.
        """
        # During inference with sampling, multi-layer blocks are passed in.
        for l, layer in enumerate(self.layers):
            y = torch.zeros(g.num_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                torch.arange(g.num_nodes()),
                sampler,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=args.num_workers)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].to(device)
                h = x[input_nodes].to(device)
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
        return y

def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (pred.argmax(1) == labels).float().sum() / len(pred)

def evaluate(model, g, inputs, labels, val_nid, test_nid, batch_size, device):
    """
    Evaluate the model on the validation and test set.
    """
    model.eval()
    with torch.no_grad():
        pred = model.inference(g, inputs, device)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid]), compute_acc(pred[test_nid], labels[test_nid])


def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # in MB

#### Main loop
def run(args, device, data):
    # Unpack data
    train_nid, val_nid, test_nid, in_feats, labels, n_classes, g = data

    # Create PyTorch DataLoader for constructing blocks
    # Note: The Cluster-GCN implementation below creates its own data loader,
    # so this part is different from the standard GraphSAGE.

    # Define model and optimizer
    model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # === Cluster-GCN specific setup ===
    # 1. ClusterGCNSampler handles partitioning internally
    print(f"Setting up Cluster-GCN with {args.num_partitions} partitions, {args.clusters_per_batch} clusters per batch.")
    cluster_sampler = dgl.dataloading.ClusterGCNSampler(
        g,
        args.num_partitions,
        cache_path=f'./{args.dataset}_cluster_gcn_partitions.pkl' # Cache partitions for faster restarts
    )
    
    dataloader = dgl.dataloading.DataLoader(
        g,
        torch.arange(args.num_partitions), # Iterate over partition IDs
        cluster_sampler,
        batch_size=args.clusters_per_batch, # This is the number of clusters to combine per batch
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        device=device # Explicitly pass the device
    )
    
    print(f"Training with {args.num_partitions} partitions, {args.clusters_per_batch} clusters per batch.")
    # =================================

    # Training loop
    best_val_acc = 0
    best_test_acc = 0
    
    if args.output_file:
        output_file = open(args.output_file, 'w')
    else:
        output_file = None

    total_training_time = 0

    for epoch in range(args.num_epochs):
        epoch_start_time = time.time()
        model.train()
        
        total_loss = 0
        
        # Loop over the clusters
        for i, block in enumerate(dataloader):
            batch_start_time = time.time()
            
            block = block.to(device)
            batch_inputs = block.srcdata['features']
            batch_labels = block.dstdata['labels']
            
            # The forward pass computes predictions on the output nodes of the block.
            # model expects a list of blocks, so wrap the single block in a list.
            batch_pred = model([block], batch_inputs)
            
            # We only compute loss on the training nodes within the subgraph
            train_mask_in_batch = block.dstdata['train_mask']
            loss = F.cross_entropy(batch_pred[train_mask_in_batch], batch_labels[train_mask_in_batch].long())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        total_training_time += epoch_duration

        # Evaluate
        if (epoch + 1) % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                # We need to run inference on the whole graph, which is expensive.
                # For speed, we can evaluate on a subset or less frequently.
                # Here we do full evaluation.
                
                # Create a different dataloader for full-neighbor inference
                sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.num_layers)
                eval_dataloader = dgl.dataloading.DataLoader(
                    g, torch.arange(g.num_nodes()), sampler,
                    batch_size=args.batch_size,
                    shuffle=False, drop_last=False, num_workers=args.num_workers
                )
                
                # The model's forward pass needs to be adapted for inference on the whole graph
                # A simple way is to pass the full graph as a single block per layer
                full_graph_blocks = [g.to(device)] * args.num_layers
                pred = model(full_graph_blocks, g.ndata['features'].to(device))

                val_acc = compute_acc(pred[val_nid], labels[val_nid])
                test_acc = compute_acc(pred[test_nid], labels[test_nid])

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                
                mem_usage = get_memory_usage()
                log_line = (f"Epoch {epoch:05d} | Loss {total_loss / (i+1):.4f} | Val Acc {val_acc:.4f} | "
                            f"Test Acc {test_acc:.4f} (Best {best_test_acc:.4f}) | "
                            f"Time {epoch_duration:.4f}s | Mem {mem_usage:.2f}MB")
                print(log_line)
                if output_file:
                    output_file.write(log_line + '\n')


    print("="*50)
    print(f"Total training time: {total_training_time:.2f}s")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best test accuracy: {best_test_acc:.4f}")
    
    final_log = f"Final Test Accuracy {best_test_acc:.4f}"
    if output_file:
        output_file.write(final_log + '\n')
        output_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphSAGE with Cluster-GCN')
    parser.add_argument("--dataset", type=str, default="ogbn-products", help="Dataset name ('ogbn-products', 'ogbn-arxiv', 'reddit').")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--wd", type=float, default=0, help="weight decay")
    parser.add_argument("--num-epochs", type=int, default=100, help="number of training epochs")
    parser.add_argument("--num-layers", type=int, default=2, help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=128, help="number of hidden units")
    parser.add_argument("--batch-size", type=int, default=1024, help="batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=4, help="number of workers for dataloader")
    parser.add_argument("--eval-every", type=int, default=5, help="evaluate every EVAL_EVERY epochs")
    parser.add_argument("--output-file", type=str, help="File to write epoch data to")
    
    # Cluster-GCN specific arguments
    parser.add_argument("--num-partitions", type=int, default=1000, help="Number of clusters to partition the graph into.")
    parser.add_argument("--clusters-per-batch", type=int, default=20, help="Number of clusters to combine for each training batch.")

    args = parser.parse_args()

    if args.gpu < 0:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:%d' % args.gpu)

    # Load graph
    if args.dataset == 'reddit':
        g, n_classes = load_reddit()
    elif args.dataset.startswith('ogbn'):
        g, n_classes = load_ogb(args.dataset)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    in_feats = g.ndata['features'].shape[1]
    labels = g.ndata['labels'].to(device)

    # Find the node IDs in the training, validation, and test set.
    train_nid = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0].to(device)
    val_nid = torch.nonzero(g.ndata['val_mask'], as_tuple=True)[0].to(device)
    test_nid = torch.nonzero(g.ndata['test_mask'], as_tuple=True)[0].to(device)

    # Pack data
    data_packed = train_nid, val_nid, test_nid, in_feats, labels, n_classes, g
    
    run(args, device, data_packed)
