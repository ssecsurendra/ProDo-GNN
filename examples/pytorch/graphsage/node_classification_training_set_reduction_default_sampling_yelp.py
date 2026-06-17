import dgl
import argparse
import time
import matplotlib.pyplot as plt
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
# from dgl.sampling.metis_sampling import *
from sklearn.metrics import f1_score
import tqdm
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)
from ogb.nodeproppred import DglNodePropPredDataset
#import dgl.data.CoraGraphDataset
from dgl.data import CoraGraphDataset,RedditDataset,FlickrDataset,YelpDataset
import numpy as np
import cupy as cp


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        #self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        #self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, "mean"))
        #self.layers.append(dglnn.SAGEConv(in_size, hid_size, "gcn"))
        #self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "gcn"))
        #self.layers.append(dglnn.SAGEConv(hid_size, out_size, "gcn"))

        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x

        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            #print("Checking weight", block.edata['weight'])
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            #cluster_id,
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
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
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
    y_true = torch.cat(ys).cpu().numpy()
    y_pred = torch.cat(y_hats).sigmoid().cpu().numpy() > 0.5
    # y_pred = torch.cat(y_hats).sigmoid().numpy() > 0.5
    # Compute metrics
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')        
    return MF.accuracy(
        torch.cat(y_hats),
        torch.cat(ys),
        #task="multiclass",
        task="multilabel",
        num_labels=num_classes,
        threshold=0.5
    ),f1_micro,f1_macro        
    # return MF.accuracy(
    #     torch.cat(y_hats),
    #     torch.cat(ys),
    #     task="multiclass",
    #     num_classes=num_classes,
    # )


def layerwise_infer(device, graph, nid, model, num_classes, batch_size):
    model.eval()
    with torch.no_grad():
        pred = model.inference(
            graph, device, batch_size
        )  # pred in buffer_device
        pred = pred[nid]
        label = graph.ndata["label"][nid].to(pred.device)
        y_true = label.cpu().numpy()
        y_pred = pred.sigmoid().cpu().numpy() > 0.5
        f1_micro = f1_score(y_true, y_pred, average='micro')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        return MF.accuracy(
            pred, label, 
            #task="multiclass",
            task="multilabel",
            num_labels=num_classes,
            threshold=0.5
        ),f1_micro,f1_macro
        # return MF.accuracy(
            # pred, label, task="multiclass", num_classes=num_classes
        # )


def train(args, device, g, 
          # cluster_id,
          dataset, model, num_classes, centrality_vals):

    # create sampler & dataloader
    #train_idx = dataset.train_idx.to(device)
    #val_idx = dataset.val_idx.to(device)
    train_mask=g.ndata['train_mask']
    val_mask=g.ndata['val_mask']
    train_idx = torch.nonzero(train_mask).squeeze().to(device)
    # plt.savefig(plot_name, format='eps')
    # degree_vals = []
    # with open(degree_file, 'r') as f:
    #     for line in f:
    #         degree_vals.append(float(line.strip()))
    # degree_vals = torch.tensor(degree_vals)
    #
    # # Check if length matches number of nodes
    # assert len(degree_vals) == g.num_nodes(), "Mismatch between degree centrality file and graph nodes"
    #print("# training nodes: ",len(train_idx)
    # Filter training nodes by degree centrality
    # degree centrality values for training nodes
    train_degrees = centrality_vals[train_idx]  

    # sort training nodes by degree centrality (descending order)
    sorted_idx = cp.argsort(-train_degrees)  

    # how many to keep (top 70%)
    k = int(1.0 * len(train_idx))  

    # select top k training nodes
    top_train_idx = train_idx[sorted_idx[:k]]
    print("sorted training node: ",top_train_idx)
    # device = train_idx.device
    N = g.num_nodes()

    # Boolean mask for training nodes
    is_train = torch.zeros(N, dtype=torch.bool, device=device)
    is_train[train_idx] = True

    # Remaining nodes to dominate
    remaining = is_train.clone()

    dominating_set_list = []

    # Pre-fetch adjacency in CSR form (GPU-friendly)
    indptr, indices, edge_ids = g.adj_tensors('csr')

    dominating_start_time = time.time()
    # Iterate nodes in eigenvalue-centrality order
    for u in top_train_idx:
    # for u in train_idx:
        u = u.item()

        # Skip if already dominated
        if not remaining[u]:
            continue

        # Add u to dominating set
        dominating_set_list.append(u)

        # Get neighbors of u
        start, end = indptr[u], indptr[u + 1]
        nbrs = indices[start:end]

        # Keep only training neighbors
        train_nbrs = nbrs[is_train[nbrs]]
        # Mark u and its training neighbors as dominated
        remaining[u] = False
        remaining[train_nbrs] = False

        # Optional early exit
        if not remaining.any():
            break

    dominating_set = torch.tensor(dominating_set_list, device=device)
    dominating_end_time = time.time()
    print("Dominating set computation time: ",dominating_end_time - dominating_start_time)
    print("Dominating set:",dominating_set)
    print("Original training nodes:", len(train_idx))
    # print("Filtered training nodes (top 70%):", len(top_train_idx))
    print("Number of nodes in dominating set:", len(dominating_set))
    target_train_size = int(args.target_train_percentage * len(train_idx))
    final_train_idx_set = set(dominating_set_list)
    Maintaining_training_start = time.time()
    if len(final_train_idx_set) < target_train_size:
        for node in top_train_idx:
            if len(final_train_idx_set) >= target_train_size:
                break
            final_train_idx_set.add(node.item())
    maintaining_training_end = time.time()
    print("Training node maintaining time: ", maintaining_training_end - Maintaining_training_start)        
    final_train_idx = torch.tensor(list(final_train_idx_set), device=device)
    print(f"Final training nodes (Dominating Set + Top Centrality): {len(final_train_idx)}")
    val_idx = torch.nonzero(val_mask).squeeze().to(device)
    #print("# val nodes: ",len(val_idx))
    sampler_time = time.time()

    sampler = NeighborSampler(
        [int(fanout) for fanout in args.fanout.split(",")],  # fanout for [layer-0, layer-1, layer-2]
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
    )
    sampler_end_time = time.time()
    sampler_time = sampler_end_time-sampler_time
    #print("Sampler time:", sampler_time, "seconds")


    use_uva = args.mode == "mixed"
    Tdataload_time = time.time()
    train_dataloader = DataLoader(
        g,
        #top_train_idx,
        #dominating_set,
        final_train_idx,
        sampler,
        #cluster_id,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )
    #print("After train_dataloader")
    Tdataload_end_time = time.time()
    Tdataload_time = Tdataload_end_time-Tdataload_time
    #print("Tdataload time:", Tdataload_time, "seconds")
    Vdataload_time = time.time()

    val_dataloader = DataLoader(
        g,
        val_idx,
        sampler,
        #cluster_id,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )
    Vdataload_end_time = time.time()
    Vdataload_time = Vdataload_end_time-Vdataload_time
    #print("Vdataload time:", Vdataload_time, "seconds")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    total_training_time = 0.0
    total_for_loop_time = 0.0
    total_model_time = 0.0
    epoch_lines = []
    total_src_nodes_layer_3=0
    total_src_nodes_layer_2=0
    total_src_nodes_layer_1=0

    for epoch in range(args.epoch):
        model.train()
        total_loss = 0
        execution_time = 0.0
        iteration_time1 =0.0
        model_time1 =0.0
        x_y_time = 0.0
        pred_time = 0.0
        loss_time = 0.0
        backward_time = 0.0
        optim_time = 0.0
        start_time1 = time.time()
        #print("Epoch",epoch)
        iter_time = time.time()
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader
        ):
            iteration_time = time.time() - iter_time
            #print("Iteration Time ",iteration_time)
            batch_start_time = time.time()
            #print("Batch processing");

            start_x_y_time = time.time()
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]
            #print("1st layer:",blocks[0])
            #print("2nd layer:",blocks[1])
            #print("3rd layer:",blocks[-1])
            if epoch == 0:
                total_src_nodes_layer_3 = total_src_nodes_layer_3 + blocks[0].num_src_nodes()
                total_src_nodes_layer_2 = total_src_nodes_layer_2 + blocks[1].num_src_nodes()
                total_src_nodes_layer_1 = total_src_nodes_layer_1 + blocks[-1].num_src_nodes()
            end_x_y_time = time.time()

            start_pred_time = time.time()
            #print("before forward pass\n");
            y_hat = model(blocks, x)
            # print("y:",y)
            # print("Shape of y",y.shape)
            # print("y_hat:",y_hat)
            # print("Shape of y_hat:",y_hat.shape)
            end_pred_time = time.time()

            start_loss_time = time.time()
            #print("After forward pass\n");
            # loss = F.cross_entropy(y_hat, y)
            loss = F.binary_cross_entropy_with_logits(y_hat, y.float())
            end_loss_time = time.time()

            start_backward_time = time.time()
            opt.zero_grad()
            #print("before backward pass\n");
            loss.backward()
            end_backward_time = time.time()

            start_optim_time = time.time()
            #print("after backward pass\n");
            opt.step()
            end_optim_time = time.time()

            total_loss += loss.item()
            batch_end_time = time.time()
            model_time = batch_end_time - batch_start_time
            #print("model Time ", model_time)
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
            #minibatch_time = time.time()-iter_time
            #print("mini batch time ",minibatch_time)
            #print("loop time {:.5f} | model time {:.5f} | xy time {:.5f} | forward time {:.5f} | backword time {:.5f} | optim time {:.5f}".format(iteration_time, model_time, x_y_time1, pred_time1, backward_time1, optim_time1))
            iter_time = time.time()
            #batch_start_time = time.time()

        end_time1 = time.time()
        execution_time = end_time1 - start_time1
        total_training_time += execution_time
        total_for_loop_time += iteration_time1
        total_model_time += model_time1
        if epoch == 0:
            layer_line = "Layer_1 {:d} | Layer_2 {:d} | Layer_3 {:d}" .format(int(total_src_nodes_layer_1/(it+1)), int(total_src_nodes_layer_2/(it+1)), int(total_src_nodes_layer_3/(it+1)))
            epoch_lines.append(layer_line)
        acc,micro,macro = evaluate(model, g, val_dataloader, num_classes)
        #print(
         #   "\nEpoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | Time : {}\n".format(
          #       epoch, total_loss / (it + 1), acc.item(), execution_time
           #  )
        #)
        #epoch_line = "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | Time : {}".format(epoch, total_loss / (it + 1), acc.item(), execution_time )
        epoch_line = "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | Time : {:.4f} | Loop_Time : {:.4f} | Model_Time : {:.4f} | x_y_time : {:.4f} | pred_time : {:.4f} | backward_time : {:.4f} | optim_time : {:.4f} ".format(epoch, total_loss / (it + 1), acc.item(), execution_time, iteration_time1, model_time1, x_y_time, pred_time, backward_time, optim_time )
        #print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | Time : {:.4f} | Loop_Time : {:.4f} | Model_Time : {:.4f} | x_y_time : {:.4f} | pred_time : {:.4f} | backward_time : {:.4f} | optim_time : {:.4f} ".format(epoch, total_loss / (it + 1), acc.item(), execution_time, iteration_time1, model_time1, x_y_time, pred_time, backward_time, optim_time ))

        epoch_lines.append(epoch_line)
    #tt_str = "total for loop time, total model time, total_training_time"
    #tt_time = "{:.4f}, {:.4f}, {:.4f}".format(total_for_loop_time, total_model_time, total_training_time)
    tt_time = "Sampling time: {:.4f}, Model training time: {:.4f}, Total time {:.4f}".format(total_for_loop_time, total_model_time, total_training_time)
    #epoch_lines.append(tt_str)
    epoch_lines.append(tt_time)     
    #tt_time = "Total Training time {:.4f}".format( total_training_time)
    #epoch_lines.append(tt_time)
    return epoch_lines

    #print( "Total Training time {:.4f}".format( total_training_time))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="puregpu",
        choices=["cpu", "mixed", "puregpu"],
        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
        "'puregpu' for pure-GPU training.",
    )
    # parser.add_argument(
    #     "--method",
    #     default="cling",
    #     choices=["graphsage", "cling"],
    #     help="graphsage vs cling",
    #     )
    parser.add_argument(
        "--dt",
        type=str,
        default="float",
        help="data type(float, bfloat16)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="yelp",
        #help="Dataset name ('cora', 'flickr', 'reddit', 'yelp', 'ogbn-products','ogbn-arxiv').",
    )
    parser.add_argument("--fanout", type=str, default="20,20,20")
    # parser.add_argument("--num_clusters", type=str, default="20")
    #parser.add_argument("--fanout", type=str, default="20,20,20,20,20")
    #parser.add_argument("--fan_out", type=str, default="25,10")

    #parser.add_argument("--fan_out", type=str, default="15,15,15")
    parser.add_argument("--target_train_percentage", type=float, default=0.7)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epoch", type=int, default=100)
    args = parser.parse_args()
    #print(f"Training with DGL built-in GraphConv module.")

    # load and preprocess dataset
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
    g = dataset[0]
    # print("G: ",G)
    # print("ndata: ",G.ndata)
    # print("edata: ",G.edata)
    # if len(G.edata) > 0:
    #     print("G has edge feature")
    # method = get_method(method)
    test_mask=g.ndata['test_mask']
    test_idx = torch.nonzero(test_mask).squeeze()
    # print("Making graph bidirected to match centrality calculation...")
    # G = dgl.to_bidirected(G, copy_ndata=True)
    g = g.to("cuda" if args.mode == "puregpu" else "cpu")
    # Suppose g is your DGLGraph
    indptr, indices, edge_ids = g.adj_tensors('csr')
    # print("indices before: ",indices)
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")
    # --- Load binary files using np.load ---
    print("Loading binary .npy files...")
    sorted_col_idx = torch.from_numpy(np.load(sortedcol_file)).to(device)
    centrality_vals = torch.from_numpy(np.load(centrality_file)).to(device)
    print("Files loaded.")
    #print(indices.shape)
    assert sorted_col_idx.shape == indices.shape
    #Check if length matches number of nodes
    assert len(centrality_vals) == g.num_nodes(), "Mismatch between degree centrality file and graph nodes"
    if not torch.cuda.is_available():
        args.mode = "cpu"
   
    num_classes = dataset.num_classes
    labels = g.ndata["label"]
    # num_classes = int(labels.max().item()) + 1
    #num_classes = 107
    # device = torch.device("cpu" if args.mode == "cpu" else "cuda")

    # create GraphSAGE model
    in_size = g.ndata["feat"].shape[1]
    # print("Feature_dim: ",in_size)
    out_size = dataset.num_classes
    # out_size = int(labels.max().item()) + 1
    #out_size = 107
    model = SAGE(in_size, 256, out_size).to(device)

    # convert model and graph to bfloat16 if needed
    if args.dt == "bfloat16":
        g = dgl.to_bfloat16(g)
        model = model.to(dtype=torch.bfloat16)

    # model training
    #print("Training...")
    epoch_lines = train(args, device, g, 
                        # cluster_id, 
                        dataset, model, num_classes, centrality_vals)

    # test the model
    #print("Testing...")
    acc,f1_micro,f1_macro = layerwise_infer(
        device, g, test_idx, model, num_classes, batch_size=4096
    )
    #acc = layerwise_infer(
        #device, g, dataset.test_idx, model, num_classes, batch_size=4096
    #)
    #end_time = time.time()
    #execution_time = end_time - start_time
    #print("Test Accuracy {:.4f}".format(acc.item()))
    #print("Execution time:", execution_time, "seconds")
    Accuracy = "Test Accuracy {:.4f}".format(acc.item())
    #tt_time = "Total Training time {:.4f}".format( total_training_time)
    #epoch_lines.append(tt_time)
    epoch_lines.append(Accuracy)
    with open('epoch_data.txt', 'w') as file:
        for value in epoch_lines:
            file.write(str(value) + '\n')

