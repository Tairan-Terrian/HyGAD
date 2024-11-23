import networkx as nx
from spectral_machinery import WaveletMachine
import dgl
import argparse
from dgl.data.utils import load_graphs
import os
from tqdm import tqdm
import torch
def get_dataset(name: str, raw_dir: str, to_homo: bool = False, random_state: int = 717):
    if name == 'yelp':
        yelp_data = dgl.data.FraudYelpDataset(raw_dir=raw_dir, random_seed=7537, verbose=False)
        graph = yelp_data[0]
        if to_homo:
            graph = dgl.to_homogeneous(graph, ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])


    elif name == 'amazon':
        amazon_data = dgl.data.FraudAmazonDataset(raw_dir=raw_dir, random_seed=7537, verbose=False)
        graph = amazon_data[0]
        if to_homo:
            graph = dgl.to_homogeneous(graph, ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])


    elif name == 'tsocial':
        t_social, _ = load_graphs(os.path.join(raw_dir, 'tsocial'))
        graph = t_social[0]
        graph.ndata['feature'] = graph.ndata['feature'].float()

    elif name == 'tfinance':
        t_finance, _ = load_graphs(os.path.join(raw_dir, 'tfinance'))
        graph = t_finance[0]
        graph.ndata['label'] = graph.ndata['label'].argmax(1)
        graph.ndata['feature'] = graph.ndata['feature'].float()

    else:
        raise

    return graph

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GraphWave.")

    parser.add_argument("--mechanism",
                        nargs="?",
                        default="exact",
                        help="Eigenvalue calculation method. Default is exact.")

    parser.add_argument("--dataset",
                        nargs="?",
                        default="yelp",
                        help="Path to the graph edges. Default is food_edges.csv.")

    parser.add_argument("--output",
                        nargs="?",
                        default="/data/embedding.csv",
                        help="Path to the structural embedding. Default is embedding.csv.")

    parser.add_argument("--heat-coefficient",
                        type=float,
                        default=1000.0,
                        help="Heat kernel exponent. Default is 1000.0.")

    parser.add_argument("--sample-number",
                        type=int,
                        default=50,
                        help="Number of characteristic function sample points. Default is 50.")

    parser.add_argument("--approximation",
                        type=int,
                        default=100,
                        help="Number of Chebyshev approximation. Default is 100.")

    parser.add_argument("--step-size",
                        type=int,
                        default=20,
                        help="Number of steps. Default is 20.")

    parser.add_argument("--switch",
                        type=int,
                        default=100,
                        help="Number of dimensions. Default is 100.")

    parser.add_argument("--node-label-type",
                        type=str,
                        default="int",
                        help="Used for sorting index of output embedding. One of 'int', 'string', or 'float'. Default is 'int'")

    parser.add_argument("--edgelist-input",
                        action='store_true',
                        help="Use NetworkX's format for input instead of CSV. Default is False")

    args = parser.parse_args()
    edges = []
    graph = get_dataset(args.dataset, 'data/', to_homo= False,  random_state=7537)
    if args.dataset == 'yelp' or args.dataset == 'amazon':
        for e_t in graph.etypes:
            # edges = graph.edges()
            u, v = graph.edges(etype=e_t, form='uv')
            _ = torch.stack([u, v])
            edges.append(_)
        edges = torch.cat(edges, dim=1)
    elif args.dataset == 'tfinance' or args.dataset == 'tsocial':
        edges = graph.edges()
    edge_list = [[edges[0][i].item(), edges[1][i].item()] for i in tqdm(range(edges[0].shape[0]))]
    new_graph = nx.from_edgelist(edge_list)

    machine = WaveletMachine(new_graph, args)
    machine.create_embedding()
    machine.transform_and_save_embedding()