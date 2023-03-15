import sys
from itertools import chain
import jax.numpy as jnp
import jax
import torch
import jraph as jg
import haiku as hk
import optax as ox
import logging

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

G_data = jg.batch([
    jg.GraphsTuple(
        n_node=jnp.asarray([X.shape[0]]),
        n_edge=jnp.asarray([G.shape[0]]),
        nodes = X[:,x:x+1],
        edges = None,
        globals = None,
        senders = G[0,:],
        receivers = G[1,:]
    ) for x in range(X.shape[1])]
)

# Model Definition
def model(graph: jg.GraphsTuple) -> jax.Array:
    gn = jg.GraphConvolution(update_node_fn=hk.Linear(10, with_bias=False), add_self_edges=True)
    graph = gn(graph)
    graph = graph._replace(nodes=jax.nn.leaky_relu(graph.nodes))
    gn = jg.GraphConvolution(update_node_fn=hk.Linear(100, with_bias=False), add_self_edges=True)
    graph = gn(graph)
    graph = graph._replace(nodes=jax.nn.leaky_relu(graph.nodes))
    gn = jg.GraphConvolution(update_node_fn=hk.Linear(100, with_bias=False), add_self_edges=True)
    graph = gn(graph)
    graph = graph._replace(nodes=jax.nn.leaky_relu(graph.nodes))
    nodes = graph.nodes.reshape([X.shape[1], -1])
    return hk.Linear(int(groups))(nodes)

network = hk.without_apply_rng(hk.transform(model))
params = network.init(jax.random.PRNGKey(42), G_data)

@jax.jit
def pred_loss(params):
    predictions = network.apply(params, G_data)
    log_prob = jax.nn.log_softmax(predictions)
    return ox.softmax_cross_entropy_with_integer_labels(log_prob, y).sum()

opt_init, opt_update = ox.adam(1e-2)
opt_state = opt_init(params)

@jax.jit
def update(params, opt_state):
    g = jax.grad(pred_loss)(params)
    updates, opt_state = opt_update(g, opt_state)
    return ox.apply_updates(params, updates), opt_state

@jax.jit
def accuracy(params):
    decoded_nodes = network.apply(params, G_data)
    return jnp.mean(jnp.argmax(decoded_nodes, axis=1) == y)

# Training Loop
for step in range(30):
    print(f"Epoch {step} Accuracy {accuracy(params).item()}")
    params, opt_state = update(params, opt_state)
