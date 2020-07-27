import torch as th


def is_depth_accessible(reducers):
    try:
        reducers[1]
        return True
    except Exception:
        return False


class TreeModel(th.nn.Module):
    """
        Custom neural network executor.
        The module performs a given operation depth-wise.
    """
    def __init__(self, reducers):
        super().__init__()

        self.reducers = reducers
        self.depth_accessible = is_depth_accessible(reducers)


    def _reduce(self, depth_id, parent_tensor, child_tensor, edge_index):
        reducer = self.reducers

        if self.depth_accessible:
            print("!!! Depth %d" % depth_id)
            reducer = self.reducers[depth_id]

        return reducer(parent_tensor, child_tensor, edge_index)

    def forward(self, x, depth_mask, edge_index, return_attach=False):

        assert x.size(0) == depth_mask.size(0), "Expected that batch dims are equal, but dim(x) = %d != %d = dim(depth_mask)" % (x.size(0), depth_mask.size(0))
        assert x.size(0) == edge_index.size(0), "Expected that batch dims are equal, but dim(x) = %d != %d = dim(edge_index)" % (x.size(0), edge_index.size(0))

        nodes = x
        max_depth = depth_mask.max().item()

        child_mask = depth_mask == max_depth
        child_buffer = nodes[child_mask]

        assert child_buffer.size(0) == child_mask.sum().item(), "Expected that after masking the child buffer fits the mask, but %d != %d" % (child_buffer.size(0), child_mask.sum().item())
        assert child_buffer.size(1) == nodes.size(1)

        attachments = []

        for current_depth in range(max_depth - 1, -1, -1):

            parent_mask = depth_mask == current_depth
            parents = nodes[parent_mask]
            edges = edge_index[child_mask]

            child_buffer = self._reduce(current_depth, parents, child_buffer, edges)

            if isinstance(child_buffer, tuple):
                child_buffer, attach = child_buffer
                attachments += [attach]

            assert child_buffer.size(0) == parents.size(0)
            child_mask = parent_mask

        if return_attach:
            return child_buffer, attachments

        return child_buffer
