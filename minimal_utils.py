import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import os

from dgl.nn.pytorch import SAGEConv
from dgl.nn.pytorch import GraphConv
from itertools import chain


# Known chromatic numbers for specified problems (from references)
chromatic_numbers = {
    # COLOR graphs
    'jean.col': 10,
    'anna.col': 11,
    'huck.col': 11,
    'david.col': 11,
    'homer.col': 13,
    'myciel5.col': 6,
    'myciel6.col': 7,
    'queen5_5.col': 5,
    'queen6_6.col': 7,
    'queen7_7.col': 7,
    'queen8_8.col': 9,
    'queen9_9.col': 10,
    'queen8_12.col': 12,
    'queen11_11.col': 11,
    'queen13_13.col': 13,
    # Citations graphs
    'cora.cites': 5,
    'citeseer.cites': 6,
    'pubmed.cites': 8
}


def set_seed(seed):
    """
    Sets random seeds for training.

    :param seed: Integer used for seed.
    :type seed: int
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_adjacency_matrix(nx_graph, torch_device, torch_dtype):
    """
    Pre-load adjacency matrix, map to torch device

    :param nx_graph: Graph object to pull adjacency matrix for
    :type nx_graph: networkx.OrderedGraph
    :param torch_device: Compute device to map computations onto (CPU vs GPU)
    :type torch_dtype: str
    :param torch_dtype: Specification of pytorch datatype to use for matrix
    :type torch_dtype: str
    :return: Adjacency matrix for provided graph
    :rtype: torch.tensor
    """

    adj = nx.linalg.graphmatrix.adjacency_matrix(nx_graph).todense()
    adj_ = torch.tensor(adj).type(torch_dtype).to(torch_device)

    return adj_


def parse_line(file_line, node_offset):
    """
    Helper function to parse lines out of COLOR files - skips first character, which
    will be an "e" to denote an edge definition, and returns node0, node1 that define
    the edge in the line.

    :param file_line: Line to be parsed
    :type file_line: str
    :param node_offset: How much to add to account for file numbering (i.e. offset by 1)
    :type node_offset: int
    :return: Set of nodes connected by edge defined in the line (i.e. node_from, node_to)
    :rtype: int, int
    """

    x, y = file_line.split(' ')[1:]  # skip first character - specifies each line is an edge definition
    x, y = int(x)+node_offset, int(y)+node_offset  # nodes in file are 1-indexed, whereas python is 0-indexed
    return x, y


def build_graph_from_color_file(fname, node_offset=-1, parent_fpath=''):
    """
    Load problem definition (graph) from COLOR file (e.g. *.col).

    :param fname: Filename of COLOR file
    :type fname: str
    :param node_offset: How much to offset node values contained in file
    :type node_offset: int
    :param parent_fpath: Path to prepend to `fname`
    :type parent_fpath: str
    :return: Graph defined in provided file
    :rtype: networkx.OrderedGraph
    """

    fpath = os.path.join(parent_fpath, fname)

    print(f'Building graph from contents of file: {fpath}')
    with open(fpath, 'r') as f:
        content = f.read().strip()

    # Identify where problem definition starts.
    # All lines prior to this are assumed to be miscellaneous descriptions of file contents
    # which start with "c ".
    start_idx = [idx for idx, line in enumerate(content.split('\n')) if line.startswith('p')][0]
    lines = content.split('\n')[start_idx:]  # skip comment line(s)
    edges = [parse_line(line, node_offset) for line in lines[1:] if len(line) > 0]

    nx_temp = nx.from_edgelist(edges)

    nx_graph = nx.OrderedGraph()
    nx_graph.add_nodes_from(sorted(nx_temp.nodes()))
    nx_graph.add_edges_from(nx_temp.edges)

    return nx_graph


# Define GNN GraphSage object
class GNNSage(nn.Module):
    """
    Basic GraphSAGE-based GNN class object. Constructs the model architecture upon
    initialization. Defines a forward step to include relevant parameters - in this
    case, just dropout.
    """

    def __init__(self, g, in_feats, hidden_size, num_classes, dropout, agg_type='mean'):
        """
        Initialize the model object. Establishes model architecture and relevant hypers (`dropout`, `num_classes`, `agg_type`)

        :param g: Input graph object
        :type g: dgl.DGLHeteroGraph
        :param in_feats: Size (number of nodes) of input layer
        :type in_feats: int
        :param hidden_size: Size of hidden layer
        :type hidden_size: int
        :param num_classes: Size of output layer (one node per class)
        :type num_classes: int
        :param dropout: Dropout fraction, between two convolutional layers
        :type dropout: float
        :param agg_type: Aggregation type for each SAGEConv layer. All layers will use the same agg_type
        :type agg_type: str
        """
        
        super(GNNSage, self).__init__()

        self.g = g
        self.num_classes = num_classes
        
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(SAGEConv(in_feats, hidden_size, agg_type, activation=F.relu))
        # output layer
        self.layers.append(SAGEConv(hidden_size, num_classes, agg_type))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        """
        Define forward step of netowrk. In this example, pass inputs through convolution, apply relu
        and dropout, then pass through second convolution.

        :param features: Input node representations
        :type features: torch.tensor
        :return: Final layer representation, pre-activation (i.e. class logits)
        :rtype: torch.tensor
        """
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)

        return h


# Define GNN GraphConv object
class GNNConv(nn.Module):
    """
    Basic GraphConv-based GNN class object. Constructs the model architecture upon
    initialization. Defines a forward step to include relevant parameters - in this
    case, just dropout.
    """
    
    def __init__(self, g, in_feats, hidden_size, num_classes, dropout):
        """
        Initialize the model object. Establishes model architecture and relevant hypers (`dropout`, `num_classes`, `agg_type`)

        :param g: Input graph object
        :type g: dgl.DGLHeteroGraph
        :param in_feats: Size (number of nodes) of input layer
        :type in_feats: int
        :param hidden_size: Size of hidden layer
        :type hidden_size: int
        :param num_classes: Size of output layer (one node per class)
        :type num_classes: int
        :param dropout: Dropout fraction, between two convolutional layers
        :type dropout: float
        """
        
        super(GNNConv, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, hidden_size, activation=F.relu))
        # output layer
        self.layers.append(GraphConv(hidden_size, num_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        """
        Define forward step of netowrk. In this example, pass inputs through convolution, apply relu
        and dropout, then pass through second convolution.

        :param features: Input node representations
        :type features: torch.tensor
        :return: Final layer representation, pre-activation (i.e. class logits)
        :rtype: torch.tensor
        """
            
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h


# Construct graph to learn on #
def get_gnn(g, n_nodes, gnn_hypers, opt_params, torch_device, torch_dtype):
    """
    Helper function to load in GNN object, optimizer, and initial embedding layer.

    :param n_nodes: Number of nodes in graph
    :type n_nodes: int
    :param gnn_hypers: Hyperparameters to provide to GNN constructor
    :type gnn_hypers: dict
    :param opt_params: Hyperparameters to provide to optimizer constructor
    :type opt_params: dict
    :param torch_device: Compute device to map computations onto (CPU vs GPU)
    :type torch_dtype: str
    :param torch_dtype: Specification of pytorch datatype to use for matrix
    :type torch_dtype: str
    :return: Initialized GNN instance, embedding layer, initialized optimizer instance
    :rtype: GNN_Conv or GNN_SAGE, torch.nn.Embedding, torch.optim.AdamW
    """

    try:
        print(f'Function get_gnn(): Setting seed to {gnn_hypers["seed"]}')
        set_seed(gnn_hypers['seed'])
    except KeyError:
        print('!! Function get_gnn(): Seed not specified in gnn_hypers object. Defaulting to 0 !!')
        set_seed(0)

    model = gnn_hypers['model']
    dim_embedding = gnn_hypers['dim_embedding']
    hidden_dim = gnn_hypers['hidden_dim']
    dropout = gnn_hypers['dropout']
    number_classes = gnn_hypers['number_classes']
    agg_type = gnn_hypers['layer_agg_type'] or 'mean'

    # instantiate the GNN
    print(f'Building {model} model...')
    if model == "GraphConv":
        net = GNNConv(g, dim_embedding, hidden_dim, number_classes, dropout)
    elif model == "GraphSAGE":
        net = GNNSage(g, dim_embedding, hidden_dim, number_classes, dropout, agg_type)
    else:
        raise ValueError("Invalid model type input! Model type has to be in one of these two options: ['GraphConv', 'GraphSAGE']")

    net = net.type(torch_dtype).to(torch_device)
    embed = nn.Embedding(n_nodes, dim_embedding)
    embed = embed.type(torch_dtype).to(torch_device)

    # set up Adam optimizer
    params = chain(net.parameters(), embed.parameters())

    print('Building ADAM-W optimizer...')
    optimizer = torch.optim.AdamW(params, **opt_params, weight_decay=1e-2)

    return net, embed, optimizer


# helper function for graph-coloring loss
def loss_func_mod(probs, adj_tensor):
    """
    Function to compute cost value based on soft assignments (probabilities)

    :param probs: Probability vector, of each node belonging to each class
    :type probs: torch.tensor
    :param adj_tensor: Adjacency matrix, containing internode weights
    :type adj_tensor: torch.tensor
    :return: Loss, given the current soft assignments (probabilities)
    :rtype: float
    """

    # Multiply probability vectors, then filter via elementwise application of adjacency matrix.
    #  Divide by 2 to adjust for symmetry about the diagonal
    loss_ = torch.mul(adj_tensor, (probs @ probs.T)).sum() / 2

    return loss_


# helper function for custom loss according to Q matrix
def loss_func_color_hard(coloring, nx_graph):
    """
    Function to compute cost value based on color vector (0, 2, 1, 4, 1, ...)

    :param coloring: Vector of class assignments (colors)
    :type coloring: torch.tensor
    :param nx_graph: Graph to evaluate classifications on
    :type nx_graph: networkx.OrderedGraph
    :return: Cost of provided class assignments
    :rtype: torch.tensor
    """

    cost_ = 0
    for (u, v) in nx_graph.edges:
        cost_ += 1*(coloring[u] == coloring[v])*(u != v)

    return cost_


def run_gnn_training(nx_graph, graph_dgl, adj_mat, net, embed, optimizer,
                     number_epochs=int(1e5), patience=1000, tolerance=1e-4, seed=1):
    """
    Function to run model training for given graph, GNN, optimizer, and set of hypers.
    Includes basic early stopping criteria. Prints regular updates on progress as well as
    final decision.

    :param nx_graph: Graph instance to solve
    :param graph_dgl: Graph instance to solve
    :param adj_mat: Adjacency matrix for provided graph
    :type adj_mat: torch.tensor
    :param net: GNN instance to train
    :type net: GNN_Conv or GNN_SAGE
    :param embed: Initial embedding layer
    :type embed: torch.nn.Embedding
    :param optimizer: Optimizer instance used to fit model parameters
    :type optimizer: torch.optim.AdamW
    :param number_epochs: Limit on number of training epochs to run
    :type number_epochs: int
    :param patience: Number of epochs to wait before triggering early stopping
    :type patience: int
    :param tolerance: Minimum change in cost to be considered non-converged (i.e.
        any change less than tolerance will add to early stopping count)
    :type tolerance: float

    :return: Final model probabilities, best color vector found during training, best loss found during training,
    final color vector of training, final loss of training, number of epochs used in training
    :rtype: torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor, int
    """

    # Ensure RNG seeds are reset each training run
    print(f'Function run_gnn_training(): Setting seed to {seed}')
    set_seed(seed)

    inputs = embed.weight

    # Tracking
    best_cost = torch.tensor(float('Inf'))  # high initialization
    best_loss = torch.tensor(float('Inf'))
    best_coloring = None

    # Early stopping to allow NN to train to near-completion
    prev_loss = 1.  # initial loss value (arbitrary)
    cnt = 0  # track number times early stopping is triggered

    # Training logic
    for epoch in range(number_epochs):

        # get soft prob assignments
        logits = net(inputs)

        # apply softmax for normalization
        probs = F.softmax(logits, dim=1)

        # get cost value with POTTS cost function
        loss = loss_func_mod(probs, adj_mat)

        # get cost based on current hard class assignments
        # update cost if applicable
        coloring = torch.argmax(probs, dim=1)
        cost_hard = loss_func_color_hard(coloring, nx_graph)

        if cost_hard < best_cost:
            best_loss = loss
            best_cost = cost_hard
            best_coloring = coloring

        # Early stopping check
        # If loss increases or change in loss is too small, trigger
        if (abs(loss - prev_loss) <= tolerance) | ((loss - prev_loss) > 0):
            cnt += 1
        else:
            cnt = 0
        
        # update loss tracking
        prev_loss = loss

        if cnt >= patience:
            print(f'Stopping early on epoch {epoch}. Patience count: {cnt}')
            break

        # run optimization with backpropagation
        optimizer.zero_grad()  # clear gradient for step
        loss.backward()  # calculate gradient through compute graph
        optimizer.step()  # take step, update weights

        # tracking: print intermediate loss at regular interval
        if epoch % 1000 == 0:
            print('Epoch %d | Soft Loss: %.5f' % (epoch, loss.item()))
            print('Epoch %d | Discrete Cost: %.5f' % (epoch, cost_hard.item()))

    # Print final loss
    print('Epoch %d | Final loss: %.5f' % (epoch, loss.item()))
    print('Epoch %d | Lowest discrete cost: %.5f' % (epoch, best_cost))

    # Final coloring
    final_loss = loss
    final_coloring = torch.argmax(probs, 1)
    print(f'Final coloring: {final_coloring}, soft loss: {final_loss}')

    return probs, best_coloring, best_loss, final_coloring, final_loss, epoch