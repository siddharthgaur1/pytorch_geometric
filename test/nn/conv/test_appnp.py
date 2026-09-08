import pytest
import torch
from torch import Tensor

import torch_geometric.typing
from torch_geometric.nn import APPNP
from torch_geometric.testing import is_full_test
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import to_edge_index, to_torch_csc_tensor


def test_appnp():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    adj1 = to_torch_csc_tensor(edge_index, size=(4, 4))

    conv = APPNP(K=3, alpha=0.1, cached=True)
    assert str(conv) == 'APPNP(K=3, alpha=0.1)'
    out = conv(x, edge_index)
    assert out.size() == (4, 16)
    assert torch.allclose(conv(x, adj1.t()), out, rtol=1e-5, atol=1e-6)
    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert torch.allclose(conv(x, adj2.t()), out, rtol=1e-5, atol=1e-6)

    # Run again to test the cached functionality:
    assert conv._cached_edge_index is not None
    assert torch.allclose(conv(x, edge_index), conv(x, adj1.t()), rtol=1e-5,
                          atol=1e-6)
    if torch_geometric.typing.WITH_TORCH_SPARSE:
        assert conv._cached_adj_t is not None
        assert torch.allclose(conv(x, edge_index), conv(x, adj2.t()),
                              rtol=1e-5, atol=1e-6)

    conv.reset_parameters()
    assert conv._cached_edge_index is None
    assert conv._cached_adj_t is None

    if is_full_test():
        jit = torch.jit.script(conv)
        assert torch.allclose(jit(x, edge_index), out, rtol=1e-5, atol=1e-6)

        if torch_geometric.typing.WITH_TORCH_SPARSE:
            assert torch.allclose(jit(x, adj2.t()), out, rtol=1e-5, atol=1e-6)


def test_appnp_dropout():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    adj1 = to_torch_csc_tensor(edge_index, size=(4, 4))

    # With dropout probability of 1.0, the final output equals to alpha * x:
    conv = APPNP(K=2, alpha=0.1, dropout=1.0)
    assert torch.allclose(0.1 * x, conv(x, edge_index), rtol=1e-5, atol=1e-6)
    assert torch.allclose(0.1 * x, conv(x, adj1.t()), rtol=1e-5, atol=1e-6)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert torch.allclose(0.1 * x, conv(x, adj2.t()), rtol=1e-5, atol=1e-6)


def _propagated_edge_weights(conv, x, edge_index):
    """The edge weights APPNP actually propagates with, one tensor per step."""
    per_step = []

    def hook(module, inputs):
        adj, kwargs = inputs[0], inputs[-1]
        if isinstance(adj, Tensor) and adj.is_sparse_csr:
            _, value = to_edge_index(adj)
        elif isinstance(adj, SparseTensor):
            value = adj.storage.value()
        else:
            value = kwargs['edge_weight']
        per_step.append(value.detach().clone())

    handle = conv.register_propagate_forward_pre_hook(hook)
    try:
        conv(x, edge_index)
    finally:
        handle.remove()

    return per_step


@pytest.mark.parametrize('adj_type', ['edge_index', 'torch_sparse', 'sparse'])
def test_appnp_dropout_is_resampled_each_step(adj_type):
    """Edge dropout must be drawn afresh at every propagation step.

    Applying it to the previous step's already-thinned weights
    compounds the mask, so an edge survives with probability
    (1 - p) ** K instead of (1 - p), and the 1 / (1 - p) rescaling
    stacks on the few survivors.
    """
    if adj_type == 'sparse' and not torch_geometric.typing.WITH_TORCH_SPARSE:
        pytest.skip('torch-sparse not installed')

    num_nodes, num_edges = 64, 512
    torch.manual_seed(12345)
    x = torch.randn(num_nodes, 8)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))

    if adj_type == 'edge_index':
        adj = edge_index
    elif adj_type == 'torch_sparse':
        adj = to_torch_csc_tensor(edge_index, size=(num_nodes, num_nodes)).t()
    else:
        adj = SparseTensor.from_edge_index(edge_index,
                                           sparse_sizes=(num_nodes,
                                                         num_nodes)).t()

    conv = APPNP(K=4, alpha=0.1, dropout=0.5)
    per_step = _propagated_edge_weights(conv, x, adj)
    assert len(per_step) == 4

    surviving = [w != 0 for w in per_step]

    # Compounding shows up as strict nesting: every later step's
    # survivors are a subset of the previous step's.
    for step, (previous, current) in enumerate(zip(surviving, surviving[1:])):
        revived = int((current & ~previous).sum())
        assert revived > 0, (
            f'step {step + 1} kept no edge that step {step} had '
            f'dropped, so dropout was applied on top of the previous '
            f'step rather than resampled')

    # Each step keeps roughly half the edges, not 1 / 2 ** k of them.
    for step, mask in enumerate(surviving):
        kept = int(mask.sum())
        assert 0.3 < kept / mask.numel() < 0.7, (
            f'step {step} kept {kept}/{mask.numel()} edges')


def test_appnp_dropout_leaves_input_untouched():
    """A forward pass must not consume the weights it was handed."""
    num_nodes, num_edges = 32, 256
    torch.manual_seed(12345)
    x = torch.randn(num_nodes, 8)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_weight = torch.rand(num_edges)
    expected = edge_weight.clone()

    conv = APPNP(K=4, alpha=0.1, dropout=0.5, normalize=False)
    conv(x, edge_index, edge_weight)

    assert torch.equal(edge_weight, expected)
