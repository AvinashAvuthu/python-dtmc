import networkx as nx
import numpy as np

from functools import reduce, partial, wraps
from bidict import bidict

from collections import Collection, ChainMap
from math import gcd

from operator import mul

from itertools import islice
import quadprog


def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None):
    qp_G = .5 * (P + P.T)   # make sure P is symmetric
    qp_a = -q
    if A is not None:
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]


def _gather_dicts(bases, namespace):
    return dict(ChainMap(*([base.__dict__ for base in list(bases)] + [namespace]))).values()


def _generate_methods(methods, wrapper, marker, namer):
    marked = [m for m in methods if hasattr(m, marker)]
    new_methods = list(map(wrapper, marked))
    for new_method, old_method in zip(new_methods, marked):
        new_method.__name__ = namer(old_method.__name__)
        new_method.__doc__ = old_method.__doc__  # TODO: How should this manifest?
    return new_methods


def make_public(name):
    return name.lstrip('_')


def _add_methods_to_class(klass, methods):
    for method in methods:
        setattr(klass, method.__name__, method)


# ----- Labelled Method Generation Magic ---- #
class IndexMethodGenerator(type):
    def __new__(mcs, name, bases, namespace, **kwargs):
        klass = type.__new__(mcs, name, bases, dict(namespace))
        methods = _gather_dicts(bases, namespace)

        # Don't change any of the methods, just make public versions that alias the private ones
        labelled_methods = _generate_methods(methods, lambda x: x, '_labelled', make_public)
        translate_methods = _generate_methods(methods, lambda x: x, '_translate_labels', make_public)
        _add_methods_to_class(klass, labelled_methods + translate_methods)

        return klass


class LabelledMethodGenerator(IndexMethodGenerator):  # Python doesn't like this if it doesn't inherit
    def __new__(mcs, name, bases, namespace, **kwargs):
        klass = type.__new__(mcs, name, bases, dict(namespace))
        methods = _gather_dicts(bases, namespace)

        # Generate labelled methods and translate returned indices to labels
        labelled_methods = _generate_methods(methods, make_labelled, '_labelled', make_public)
        translate_methods = _generate_methods(methods, make_indexed, '_translate_labels', make_public)
        _add_methods_to_class(klass, labelled_methods + translate_methods)

        return klass


# ---- Decorators ----

def generate_labelled_method(method):
    method._labelled = True
    return method


def translate_labels(method):
    method._translate_labels = True
    return method


def map_into(f, x):
    if isinstance(x, Collection) and not isinstance(x, str):
        return type(x)(map(partial(map_into, f), x))
    return f(x)


def make_labelled(index_based):
    @wraps(index_based)
    def wrapper(self, *args, **kwargs):
        result = index_based(self, *args, **kwargs)
        return map_into(lambda x: self._labels.inv[x], result)
    return wrapper


def make_indexed(label_based):
    @wraps(label_based)
    def wrapper(self, label_data, *args, **kwargs):
        indices = map_into(lambda x: self._labels[x], label_data)
        return label_based(self, indices, *args, **kwargs)
    return wrapper


# ----- Matrix checks ----- #
def _assert_square_matrix(matrix: np.matrix) -> None:
    if max(matrix.shape) != min(matrix.shape) or matrix.ndim != 2:
        raise ValueError("The matrix must be square. Got shape {}".format(matrix.shape))


def _assert_non_negative_matrix(matrix: np.matrix) -> None:
    if np.any(matrix < 0):
        raise ValueError("The matrix can only contain non-negative values.")


def _assert_right_stochastic_matrix(matrix: np.matrix, tolerance: float = 1e-8) -> None:
    _assert_square_matrix(matrix)
    _assert_non_negative_matrix(matrix)
    row_sum = np.sum(matrix, 1)
    if not np.allclose(row_sum, np.ones(matrix.shape[0]), atol=tolerance):
        raise ValueError("The matrix must be a right stochastic matrix.")


def _reachable_from_component(graph, component):
    reachable = partial(nx.descendants, graph)
    return reduce(set.union, map(reachable, component), set())


def _is_recurrent_component(graph, component):
    return _reachable_from_component(graph, component) == set(component)


def _is_transient_component(graph, component):
    return _reachable_from_component(graph, component) - set(component) != set()


class DTMC(object, metaclass=IndexMethodGenerator):
    # """@DynamicAttrs"""
    def __init__(self, transition_matrix):
        self._p = np.asmatrix(transition_matrix)
        _assert_right_stochastic_matrix(self._p)
        self._n = self._p.shape[0]
        self._graph = nx.DiGraph(self._p)
        self._labels = None

    def closest_reversible(self, dist, weighted=False):
        dist = np.asarray(dist, dtype='float64')
        n = self._n
        if dist.size != self._n:
            raise ValueError("The initial distribution must have size equal to the number of states.")
        if np.any(dist < 0):
            raise ValueError("The dist must be non negative.")
        if weighted and np.any(dist == 0):
            raise ValueError("The dist cannot have any zeros in order to allow the reweighting scheme.")

        # Compute the number of basis vectors
        temp_b = np.sum(dist == 0)
        m = (n - 1) * n/2 + 1 + (temp_b - 1) * temp_b/2

        # The `basis` is an list of `m` matrices. It contains the vectors of the subspace U
        basis = []

        for r in range(n - 1):
            for s in range(r, n):
                if dist[s] == 0 and dist[r] == 0:
                    b1 = np.eye(n)

                    b1[r, r], b1[r, s] = 0, 1
                    basis.append(b1)  # NOTE: Maybe this is supposed to be a copy??

                    b2 = np.eye(n)
                    b2[r, r], b2[r, s], b2[s, s], b2[s, r] = 1, 0, 0, 1
                    basis.append(b2)
                else:
                    b = np.eye(n)
                    b[r, s], b[s, r], b[r, r], b[s, s] = dist[s], dist[r], 1 - dist[s], 1 - dist[r]
                    basis.append(b)
        # The final basis vector is just the identity matrix
        basis.append(np.eye(n))

        f = np.zeros((m, 1))
        Q = np.zeros((m, m))

        if not weighted:  # TODO: It looks like these loops can get factored, but tests first
            # Compute f
            for i, b in enumerate(basis):
                f[i, :] = -2*np.trace(b.conj().T*self._p)
            # Compute Q
            for i, b in enumerate(basis):
                for j, h in enumerate(basis):
                    t = 2 * (b.T * h)  # NOTE: This doesn't make any sense but fuck it
                    Q[i, j] = t  # Question: How is this shape working?
                    Q[j, i] = t
        else:
            D = np.diag(dist)
            Di = np.linalg.inv(D)
            for i, b in enumerate(basis):
                f[i] = -2 * np.trace(D*b*Di*self._p.T)

            for i, b in enumerate(basis):
                Z = D * b * Di
                for j, h in enumerate(basis):
                    t = 2 * h.T * Z   # NOTE: This doesn't make any sense but fuck it
                    Q[i, j] = t
                    Q[j, i] = t

        C = -1 * np.eye(m - 1 + n, m)
        C[m - 1, m - 1] = 0  # NOTE: The -1's here may be incorrect but i think we good

        for i in range(n):
            index = 0
            for r in range(n - 1):
                for s in range(r, n):
                    if dist[s] == 0 and dist[r] == 0:
                        C[m - 1 + i, index] = -int(r != i)
                        index += 1
                        C[m - 1 + i, index] = -int(s != i)
                    elif s == i:    # TODO:  These cases can be sauced with some special boolean multiplication
                        C[m - i + 1, index] = dist[r] - 1
                    elif r == i:
                        C[m - 1 + i, index] = dist[s] - 1
                    else:
                        C[m - 1 + i, index] = -1
                    index += 1
            C[m - 1 + i, m] = -1

        zero_vector = np.zeros((m - 1 + n, 1))
        one_vector = np.ones((1, m))

        sol = quadprog_solve_qp(Q, f, C, zero_vector, one_vector, 1)

        U = np.zeros((n, n))
        for i, b in enumerate(basis):
            U += sol[i]*b

        # TODO: Finish this bullshit

    def is_irreducible(self):
        """Check if the chain is reducible."""
        return nx.number_strongly_connected_components(self._graph) == 1

    def is_reducible(self):
        return not self.is_irreducible()

    def is_ergodic(self):
        m = (self._n - 1)**2 + 1
        return np.all(np.linalg.matrix_power(self._p, m) > 0)

    def mixing_time(self):
        w, _ = np.linalg.eig(self._p)
        u = np.sort(np.absolute(w))[-2]
        return -1 / np.log(u)

    @generate_labelled_method
    def _communicating_classes(self):
        """Return the list of communicating classes."""
        return list(nx.strongly_connected_components(self._graph))

    @generate_labelled_method
    def _recurrent_classes(self):
        """Return a list of recurrent classes."""
        is_recurrent = partial(_is_recurrent_component, self._graph)
        return list(filter(is_recurrent, self._communicating_classes()))

    @generate_labelled_method
    def _transient_classes(self):
        """Return a list of transient classes."""
        is_transient = partial(_is_transient_component, self._graph)
        return list(filter(is_transient, self._communicating_classes()))

    @generate_labelled_method
    def _absorbing_states(self):
        # If a row has one non-zero entry, it must be a one and the state is recurring
        mask = np.count_nonzero(np.asarray(self._p), 1) == 1
        return list(np.arange(self._n)[mask])

    @generate_labelled_method
    def _transient_states(self):
        """Return a list of transient states, if any."""
        return reduce(set.union, self._transient_classes(), set())

    @generate_labelled_method
    def _recurrent_states(self):
        """Return a list of recurrent states, if any."""
        return reduce(set.union, self._recurrent_classes(), set())

    @translate_labels
    def _sub_matrix(self, indices):
        return self._p[np.ix_(indices, indices)]

    @translate_labels
    def _sub_chain(self, states):
        sub_matrix = self._sub_matrix(states)
        return dtmc(sub_matrix, self._labels)  # TODO: How tf do we know if this has labels

    @translate_labels
    def _period(self, state):
        """Return the period of the given state."""
        cycles = nx.simple_cycles(self._graph)
        cycles_on_state = filter(lambda cycle: state in cycle, cycles)
        cycle_lengths = map(len, cycles_on_state)
        return reduce(gcd, cycle_lengths)
    # TODO: Calculate the cyclic classes (see http://math.bme.hu/~nandori/Virtual_lab/stat/markov/Periodicity.pdf)

    def redistribute(self, num_steps, initial_distribution=None):
        if num_steps < 1:
            raise ValueError("The number of steps must be a positive integer. Received {}".format(num_steps))
        if initial_distribution is None:
            initial_distribution = np.ones(self._n, dtype='float64') / self._n
        else:
            initial_distribution = np.asarray(initial_distribution, dtype='float64')
            if initial_distribution.size != self._n:
                raise ValueError("The initial distribution must have size equal to the number of states.")
            initial_distribution /= initial_distribution.sum()

        redistribution = np.zeros((num_steps + 1, self._n), dtype='float64')
        redistribution[0, :] = initial_distribution
        p_power = self._p * self._p
        for i in range(1, num_steps + 1):
            redistribution[i] = np.matmul(initial_distribution, p_power)
            p_power *= self._p
        return redistribution

    def steady_states(self):
        """Return the vector(s) of steady state(s)."""
        if self.is_reducible():
            raise NotImplementedError("Steady state distribution only implemented for irreducible chains.")

        n = self._n
        p = self._p.T - np.eye(n)

        sum_constraint = np.vstack((p, np.ones(n)))

        b = np.zeros(n + 1)
        b[-1] = 1

        return np.linalg.lstsq(sum_constraint, b, rcond=None)[0]


class LabelledDTMC(DTMC, metaclass=LabelledMethodGenerator):
    # """@DynamicAttrs"""
    def __init__(self, transition_matrix, labels):
        super().__init__(transition_matrix)
        if len(labels) != self._n:
            raise ValueError("The number of labels given must equal the number of items in the transition matrix." +
                             "Got {} labels for a matrix with {} states.".format(len(labels), self._n))
        if len(set(labels)) < len(labels):
            raise ValueError("The labels must be unique.")
        self._labels = bidict(zip(labels, range(self._n)))


def windows(seq, n=2):
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem, )
        yield result


def is_reversible(p: np.matrix, graph: nx.DiGraph):
    # Kolmogorov's criterion
    for cycle in nx.simple_cycles(graph):
        forward_p = reduce(mul, (p[t] for t in windows(cycle)))
        reverse_p = reduce(mul, (p[t] for t in windows(reversed(cycle))))
        if forward_p != reverse_p:
            return False
    return True


def steady_state(p):
    n = p.shape[0]
    e = np.ones((n, 1))
    ct = p.sum(0)
    hit = (np.identity(n) - p + e*ct)
    return np.linalg.solve(hit.T, ct.T).T.A[0]


def mfpt(p):
    # Note: assumes irreducible i think

    n = p.shape[0]
    pi = steady_state(p)
    e = np.ones((n, 1))
    ct = p.sum(0)
    h = (np.identity(n) - p - e*ct).I
    m = np.zeros_like(p)
    for i in range(n):
        for j in range(n):
            if i == j:
                m[i, j] = 1 / pi[j]
            else:
                m[i, j] = (h[j, j] - h[i, j]) / pi[j]
    return m


mfpt(np.asmatrix([[0.9, 0.075, 0.025], [0.15, 0.8, 0.05], [0.25, 0.25, 0.5]]))

# We need a dag of properties and derived behavior

def dtmc(transition_matrix, labels=None, row_sum_tolerance=1e-08):
    if labels is None:
        return DTMC(transition_matrix)
    return LabelledDTMC(transition_matrix, labels)
