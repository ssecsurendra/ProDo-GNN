import dgl
import argparse
import time
import matplotlib.pyplot as plt
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import tqdm
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import CoraGraphDataset,RedditDataset,FlickrDataset,YelpDataset
import numpy as np
import cupy as cp


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, "mean"))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size):
        feat = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(),
                self.hid_size if l != len(self.layers) - 1 else self.out_size,
                dtype=feat.dtype,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                y[output_nodes[0] : output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y


def evaluate(model, graph, dataloader, num_classes):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata["feat"]
            ys.append(blocks[-1].dstdata["label"])
            y_hats.append(model(blocks, x))
    return MF.accuracy(
        torch.cat(y_hats),
        torch.cat(ys),
        task="multiclass",
        num_classes=num_classes,
    )


def layerwise_infer(device, graph, nid, model, num_classes, batch_size):
    model.eval()
    with torch.no_grad():
        pred = model.inference(graph, device, batch_size)
        pred = pred[nid]
        label = graph.ndata["label"][nid].to(pred.device)
        return MF.accuracy(pred, label, task="multiclass", num_classes=num_classes)


def train(args, device, g, dataset, model, num_classes, centrality_vals):
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    train_idx = torch.nonzero(train_mask).squeeze().to(device)

    train_degrees = centrality_vals[train_idx]
    sorted_idx = torch.argsort(-train_degrees)
    top_train_idx = train_idx[sorted_idx]

    N = g.num_nodes()
    is_train = torch.zeros(N, dtype=torch.bool, device=device)
    is_train[train_idx] = True
    remaining = is_train.clone()
    dominating_set_list = []
    indptr, indices, edge_ids = g.adj_tensors('csr')

    dominating_start_time = time.time()
    for u in top_train_idx:
        u = u.item()
        if not remaining[u]:
            continue
        dominating_set_list.append(u)
        start, end = indptr[u], indptr[u + 1]
        nbrs = indices[start:end]
        train_nbrs = nbrs[is_train[nbrs]]
        remaining[u] = False
        remaining[train_nbrs] = False
        if not remaining.any():
            break
    dominating_set = torch.tensor(dominating_set_list, device=device)
    dominating_end_time = time.time()
    print("Dominating set computation time: ", dominating_end_time - dominating_start_time)
    print("Original training nodes:", len(train_idx))
    print("Number of nodes in dominating set:", len(dominating_set))

    target_train_size = int(args.target_train_percentage * len(train_idx))
    final_train_idx_set = set(dominating_set_list)
    if len(final_train_idx_set) < target_train_size:
        for node in top_train_idx:
            if len(final_train_idx_set) >= target_train_size:
                break
            final_train_idx_set.add(node.item())
    final_train_idx = torch.tensor(list(final_train_idx_set), device=device)
    print(f"Final training nodes (Dominating Set + Top Centrality): {len(final_train_idx)}")

    val_idx = torch.nonzero(val_mask).squeeze().to(device)
    sampler = NeighborSampler(
        [int(fanout) for fanout in args.fanout.split(",")],
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
    )
    use_uva = args.mode == "mixed"
    train_dataloader = DataLoader(
        g, final_train_idx, sampler, device=device,
        batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0, use_uva=use_uva
    )
    val_dataloader = DataLoader(
        g, val_idx, sampler, device=device,
        batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0, use_uva=use_uva
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    total_training_time = 0.0
    total_for_loop_time = 0.0
    total_model_time = 0.0
    epoch_lines = []
    total_src_nodes_layer_3 = 0
    total_src_nodes_layer_2 = 0
    total_src_nodes_layer_1 = 0

    for epoch in range(args.epoch):
        model.train()
        total_loss = 0
        execution_time = 0.0
        iteration_time1 = 0.0
        model_time1 = 0.0
        x_y_time = 0.0
        pred_time = 0.0
        loss_time = 0.0
        backward_time = 0.0
        optim_time = 0.0
        start_time1 = time.time()
        iter_time = time.time()
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            iteration_time = time.time() - iter_time
            batch_start_time = time.time()
            start_x_y_time = time.time()
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]
            if epoch == 0:
                total_src_nodes_layer_3 += blocks[0].num_src_nodes()
                total_src_nodes_layer_2 += blocks[1].num_src_nodes()
                total_src_nodes_layer_1 += blocks[-1].num_src_nodes()
            end_x_y_time = time.time()
            start_pred_time = time.time()
            y_hat = model(blocks, x)
            end_pred_time = time.time()
            start_loss_time = time.time()
            loss = F.cross_entropy(y_hat, y)
            end_loss_time = time.time()
            start_backward_time = time.time()
            opt.zero_grad()
            loss.backward()
            end_backward_time = time.time()
            start_optim_time = time.time()
            opt.step()
            end_optim_time = time.time()
            total_loss += loss.item()
            batch_end_time = time.time()
            model_time = batch_end_time - batch_start_time
            iteration_time1 += iteration_time
            model_time1 += model_time
            x_y_time1 = end_x_y_time - start_x_y_time
            x_y_time += x_y_time1
            pred_time1 = end_pred_time - start_pred_time
            pred_time += pred_time1
            backward_time1 = end_backward_time - start_backward_time
            backward_time += backward_time1
            optim_time1 = end_optim_time - start_optim_time
            optim_time += optim_time1
            iter_time = time.time()

        end_time1 = time.time()
        execution_time = end_time1 - start_time1
        total_training_time += execution_time
        total_for_loop_time += iteration_time1
        total_model_time += model_time1
        if epoch == 0:
            layer_line = "Layer_1 {:d} | Layer_2 {:d} | Layer_3 {:d}".format(
                int(total_src_nodes_layer_1 / (it + 1)),
                int(total_src_nodes_layer_2 / (it + 1)),
                int(total_src_nodes_layer_3 / (it + 1))
            )
            epoch_lines.append(layer_line)
        acc = evaluate(model, g, val_dataloader, num_classes)
        epoch_line = "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | Time : {:.4f} | Loop_Time : {:.4f} | Model_Time : {:.4f} | x_y_time : {:.4f} | pred_time : {:.4f} | backward_time : {:.4f} | optim_time : {:.4f} ".format(
            epoch, total_loss / (it + 1), acc.item(), execution_time, iteration_time1, model_time1, x_y_time, pred_time, backward_time, optim_time
        )
        print(epoch_line)
        epoch_lines.append(epoch_line)

    tt_time = "Sampling time: {:.4f}, Model training time: {:.4f}, Total time {:.4f}".format(
        total_for_loop_time, total_model_time, total_training_time
    )
    epoch_lines.append(tt_time)
    return epoch_lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="puregpu", choices=["cpu", "mixed", "puregpu"])
    parser.add_argument("--dt", type=str, default="float")
    parser.add_argument("--dataset", type=str, default="ogbn-arxiv")
    parser.add_argument("--fanout", type=str, default="20,20,20")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--target_train_percentage", type=float, default=0.7)
    parser.add_argument("--output-file", type=str, default="epoch_data_improved.txt")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        args.mode = "cpu"
    print(f"Training in {args.mode} mode.")

    if args.dataset == "cora":
        dataset = CoraGraphDataset()
    elif args.dataset == "citeseer":
        dataset = CiteseerGraphDataset()
    elif args.dataset == "pubmed":
        dataset = PubmedGraphDataset()
    elif args.dataset == "wisconsin":
        dataset = WisconsinDataset()
    elif args.dataset == "flickr":
        dataset = FlickrDataset()
    elif args.dataset == "reddit":
        dataset = RedditDataset()
        centrality_file = 'dissimilarity-eigenvector-centrality/reddit_dissimilar_eigen-centrality.npy'
        sortedcol_file = 'dissimilarity-eigenvector-centrality/reddit_dissimilar_eigen_sorted-col-index.npy'
    elif args.dataset == "yelp":
        dataset = YelpDataset()
        centrality_file = 'dissimilarity-eigenvector-centrality/yelp_dissimilar_eigen-centrality.npy'
        sortedcol_file = 'dissimilarity-eigenvector-centrality/yelp_dissimilar_eigen_sorted-col-index.npy'
    elif args.dataset == "ogbn-products":
        dataset = AsNodePredDataset(DglNodePropPredDataset("ogbn-products"))
        centrality_file = 'dissimilarity-eigenvector-centrality/ogbn-products_dissimilar_eigen-centrality.npy'
        sortedcol_file = 'dissimilarity-eigenvector-centrality/ogbn-products_dissimilar_eigen_sorted-col-index.npy'
    elif args.dataset == "ogbn-arxiv":
        dataset = AsNodePredDataset(DglNodePropPredDataset("ogbn-arxiv"))
        centrality_file = 'dissimilarity-eigenvector-centrality/ogbn-arxiv_dissimilar_eigen-centrality.npy'
        sortedcol_file = 'dissimilarity-eigenvector-centrality/ogbn-arxiv_dissimilar_eigen_sorted-col-index.npy'
    elif args.dataset == "amazon_products":
        load_path = '/data/Dataset/gnn_dataset/amazon_products.dgl'
        dataset, _ = dgl.load_graphs(load_path)
    elif args.dataset == "cit-net":
        load_path = '/data/Dataset/gnn_dataset/citations_network_graph.dgl'
        dataset, _ = dgl.load_graphs(load_path)
    elif args.dataset == "igb-tiny":
        load_path = './dataset/igb_tiny.dgl'
        dataset, _ = dgl.load_graphs(load_path)
    elif args.dataset == "igb-medium":
        load_path = './dataset/igb_medium.dgl'
        dataset, _ = dgl.load_graphs(load_path)
    elif args.dataset == "wiki":
        load_path = './dataset/wikidata5M/wikidata5m_dgl_graph.bin'
        dataset, _ = dgl.load_graphs(load_path)
    elif args.dataset == "igb-small":
        load_path = './dataset/igb_small.dgl'
        dataset, _ = dgl.load_graphs(load_path)
        centrality_file = 'dissimilarity-eigenvector-centrality/igb-small_dissimilar_eigen-centrality.npy'
        sortedcol_file = 'dissimilarity-eigenvector-centrality/igb-small_dissimilar_eigen_sorted-col-index.npy'
    elif args.dataset == "amazon_products":
        load_path = './dataset/amazon_products.dgl'
        dataset, _ = dgl.load_graphs(load_path)       
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))
        
    G = dataset[0]
    test_mask = G.ndata['test_mask']
    test_idx = torch.nonzero(test_mask).squeeze()
    G = G.to("cuda" if args.mode == "puregpu" else "cpu")
    indptr, indices, edge_ids = G.adj_tensors('csr')
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")
    
    print("Loading binary .npy files...")
    sorted_col_idx = torch.from_numpy(np.load(sortedcol_file)).to(device)
    centrality_vals = torch.from_numpy(np.load(centrality_file)).to(device)
    print("Files loaded.")
    
    assert sorted_col_idx.shape == indices.shape
    num_nodes = len(indptr) - 1
    num_edges = indptr[-1].item()
    orig_edge_ids = torch.arange(num_edges, device=device)

    edge_pos = torch.arange(num_edges, dtype=torch.int64, device=indptr.device)
    row_ids = torch.searchsorted(indptr[1:], edge_pos, right=True)
    col_ids = sorted_col_idx.to(torch.int64)
    
    g = dgl.graph((row_ids, col_ids), num_nodes=num_nodes)
    g.edata['__orig__'] = orig_edge_ids
    g.ndata.update(G.ndata)
    if len(G.edata) > 0:
        orig_eids = g.edata['__orig__']
        new_edata = {}
        for key, feat in G.edata.items():
            if key == '__orig__':
                continue
            new_edata[key] = feat[orig_eids]
        g.edata.update(new_edata)
    del G
    torch.cuda.empty_cache()
    
    labels = g.ndata["label"]
    num_classes = int(labels.max().item()) + 1

    in_size = g.ndata["feat"].shape[1]
    out_size = num_classes
    model = SAGE(in_size, 256, out_size).to(device)

    if args.dt == "bfloat16":
        g = dgl.to_bfloat16(g)
        model = model.to(dtype=torch.bfloat16)

    epoch_lines = train(args, device, g, dataset, model, num_classes, centrality_vals)

    acc = layerwise_infer(device, g, test_idx, model, num_classes, batch_size=4096)
    Accuracy = "Test Accuracy {:.4f}".format(acc.item())
    epoch_lines.append(Accuracy)
    
    with open(args.output_file, 'w') as file:
        for value in epoch_lines:
            file.write(str(value) + '\n')
