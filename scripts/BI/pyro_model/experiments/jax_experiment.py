import sys
from itertools import chain
import jax.numpy as jnp
import jax
import torch
import jraph as jg
import haiku as hk
import optax as ox
import logging
from tqdm import tqdm
from random import shuffle

# Load Data
groups = sys.argv[1]

X = torch.load(
    f"model/sergio/final_data/100gene-{groups}groups"
    f"-1sparsity.pt").float()
y = torch.load(
    f"model/sergio/final_data/100gene-{groups}groups"
    f"-labels.pt")
G = torch.load(
    f"model/sergio/final_data/100gene-{groups}groups"
    f"-1sparsity-compgraph.pt")
grn = torch.load(
    f"model/sergio/final_data/100gene-{groups}groups"
    f"-gt-grn.pt")

X = jnp.array(X)
y = jnp.array(y)
G = jnp.array(G)
grn = jnp.array(G)
grn_s = set(
    [(s.item(), d.item()) for s, d in zip(chain(grn[0, :], grn[1, :]), chain(grn[1, :], grn[0, :]))])
gt_grn = jnp.array([1 if (s.item(), d.item()) in grn_s else 0 for s, d in zip(G[0, :], G[1, :])])


batchsize = 10
Gs = [
    jg.GraphsTuple(
        n_node=jnp.asarray([X.shape[0]]),
        n_edge=jnp.asarray([G.shape[0]]),
        nodes = X[:,x:x+1],
        edges = None,
        globals = y[x:x+1],
        senders = G[0,:],
        receivers = G[1,:]
    ) for x in range(X.shape[1])
]
shuffle(Gs)
G_data = []
batches = int(len(Gs)/batchsize)
for i in range(batches):
    G_data.append(jg.batch(Gs[i*batchsize:(i+1)*batchsize]))


# Model Definition
def model(graph: jg.GraphsTuple) -> jax.Array:
    gn = jg.GraphConvolution(update_node_fn=hk.Linear(10), add_self_edges=True)
    graph = gn(graph)
    graph = graph._replace(nodes=jax.nn.leaky_relu(graph.nodes))
    gn = jg.GraphConvolution(update_node_fn=hk.Linear(64), add_self_edges=True)
    graph = gn(graph)
    graph = graph._replace(nodes=jax.nn.leaky_relu(graph.nodes))
    gn = jg.GraphConvolution(update_node_fn=hk.Linear(128), add_self_edges=True)
    graph = gn(graph)
    graph = graph._replace(nodes=jax.nn.leaky_relu(graph.nodes))
    gn = jg.GraphConvolution(update_node_fn=hk.Linear(64), add_self_edges=True)
    graph = gn(graph)
    graph = graph._replace(nodes=jax.nn.leaky_relu(graph.nodes))
    gn = jg.GraphConvolution(update_node_fn=hk.Linear(32), add_self_edges=True)
    graph = gn(graph)
    graph = graph._replace(nodes=jax.nn.leaky_relu(graph.nodes))
    nodes = graph.nodes.reshape([batchsize, -1])
    return hk.Linear(int(groups))(nodes)

network = hk.without_apply_rng(hk.transform(model))
params = network.init(jax.random.PRNGKey(42), G_data[0])

@jax.jit
def pred_loss(params, G):
    predictions = network.apply(params, G)
    return ox.softmax_cross_entropy_with_integer_labels(predictions, G.globals).sum()

opt_init, opt_update = ox.adam(0.0001)
opt_state = opt_init(params)

@jax.jit
def update(params, opt_state, G):
    g = jax.grad(pred_loss)(params, G)
    updates, opt_state = opt_update(g, opt_state)
    return ox.apply_updates(params, updates), opt_state

@jax.jit
def accuracy(params, G):
    decoded_nodes = network.apply(params, G)
    return jnp.mean(jnp.argmax(decoded_nodes, axis=1) == G.globals)

# Training Loop
best_acc = 0
pbar = tqdm(range(3000))
for step in pbar:
    acc = 0
    for G in G_data:
        acc += accuracy(params, G).item()
        params, opt_state = update(params, opt_state, G)
    acc /= len(G_data)
    best_acc = acc if acc > best_acc else best_acc
    pbar.set_description(f"Epoch {step} Best Accuracy {best_acc}")
