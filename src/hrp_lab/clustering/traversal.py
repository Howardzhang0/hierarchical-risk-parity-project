"""Reference and optimized linkage-tree traversals."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


IntArray = NDArray[np.intp]


def _children_from_linkage(linkage_matrix: ArrayLike) -> tuple[int, IntArray]:
    matrix = np.asarray(linkage_matrix, dtype=np.float64)
    if matrix.size == 0:
        matrix = np.empty((0, 4), dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 4:
        raise ValueError("linkage_matrix must have shape (n_assets - 1, 4)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("linkage_matrix contains NaN or infinite values")

    n_assets = matrix.shape[0] + 1
    n_nodes = 2 * n_assets - 1
    children = np.full((n_nodes, 2), -1, dtype=np.intp)
    if matrix.shape[0] == 0:
        return n_assets, children

    raw_children = matrix[:, :2]
    integer_children = np.rint(raw_children).astype(np.intp)
    if not np.array_equal(raw_children, integer_children):
        raise ValueError("linkage child identifiers must be integers")

    for row, pair in enumerate(integer_children):
        parent = n_assets + row
        left, right = int(pair[0]), int(pair[1])
        if left < 0 or right < 0 or left >= parent or right >= parent:
            raise ValueError("linkage contains an invalid child identifier")
        if left == right:
            raise ValueError("a linkage node cannot contain the same child twice")
        children[parent] = pair
    return n_assets, children


def _validate_leaf_order(order: IntArray, n_assets: int) -> IntArray:
    if order.shape != (n_assets,) or not np.array_equal(
        np.sort(order),
        np.arange(n_assets, dtype=np.intp),
    ):
        raise ValueError("linkage does not define a tree containing each leaf once")
    return order


def leaf_order_recursive(linkage_matrix: ArrayLike) -> IntArray:
    """Return left-to-right leaf order using a simple recursive reference."""

    n_assets, children = _children_from_linkage(linkage_matrix)
    root = 2 * n_assets - 2
    leaves: list[int] = []

    def visit(node: int) -> None:
        if node < n_assets:
            leaves.append(node)
            return
        left, right = children[node]
        visit(int(left))
        visit(int(right))

    visit(root)
    return _validate_leaf_order(np.asarray(leaves, dtype=np.intp), n_assets)


def leaf_order_stack(linkage_matrix: ArrayLike) -> IntArray:
    """Return leaf order with a preallocated stack and integer node indexes.

    This avoids Python recursion, ``ClusterNode`` objects, pandas reindexing,
    and repeated list expansion in the matrix quasi-diagonalization path.
    """

    n_assets, children = _children_from_linkage(linkage_matrix)
    n_nodes = 2 * n_assets - 1
    stack = np.empty(n_nodes, dtype=np.intp)
    order = np.empty(n_assets, dtype=np.intp)

    stack[0] = n_nodes - 1
    stack_pointer = 1
    output_pointer = 0

    while stack_pointer:
        stack_pointer -= 1
        node = int(stack[stack_pointer])
        if node < n_assets:
            if output_pointer >= n_assets:
                raise ValueError("linkage produces too many leaves")
            order[output_pointer] = node
            output_pointer += 1
            continue

        left, right = children[node]
        if left < 0 or right < 0:
            raise ValueError("linkage references an undefined internal node")
        # LIFO: push right first so that left is visited first.
        stack[stack_pointer] = right
        stack[stack_pointer + 1] = left
        stack_pointer += 2

    if output_pointer != n_assets:
        raise ValueError("linkage produces too few leaves")
    return _validate_leaf_order(order, n_assets)


quasi_diagonal_order = leaf_order_stack
