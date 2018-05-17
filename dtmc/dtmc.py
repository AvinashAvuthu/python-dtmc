# -*- coding: utf-8 -*-

"""Main module."""

from functools import reduce, partial
try:
    from math import gcd
except ImportError:
    from fractions import gcd
import networkx as nx
import numpy as np
from bidict import bidict


def _reachable_from_component(graph, component):
    reachable = partial(nx.descendants, graph)
    return reduce(set.union, map(reachable, component), set())


def _is_recurrent_component(graph, component):
    return _reachable_from_component(graph, component) == set(component)


def _is_transient_component(graph, component):
    return _reachable_from_component(graph, component) - set(component) != set()


class DiscreteTimeMarkovChain(object):
    def __init__(self, transition_matrix, labels=None, absolute_sum_tolerance=1e-08):
        self._P = np.asmatrix(transition_matrix, 'float64')
        self._num_states = self._P.shape[0]

        # Check that we got a square matrix
        if self._P.shape[0] != self._P.shape[1]:
            raise ValueError("The transition matrix must be square. Got shape {}.".format(self._P.shape))

        # Check that we got a right stochastic matrix
        if np.any(self._P < 0):
            raise ValueError("The transition matrix can only contain non-negative values.")
        if not np.allclose(np.sum(self._P, 1), np.ones(self._num_states), atol=absolute_sum_tolerance):
            raise ValueError("The transition matrix must have rows that sum to (almost) 1.")

        if labels is None:
            self.labels = bidict(zip(range(self._num_states), range(self._num_states)))
        else:
            if len(labels) != self._num_states:
                raise ValueError("The number of labels given must equal the number of items in the transition matrix." +
                                 "Got {} labels for a matrix with {} states.".format(len(labels), self._num_states))
            if len(set(labels)) != len(labels):
                raise ValueError("The labels must be unique.")
            self.labels = bidict(zip(labels, range(self._num_states)))

        self._graph = nx.DiGraph(self._P)

    def communicating_classes(self):
        """Return the list of communicating classes."""
        return list(nx.strongly_connected_components(self._graph))

    def recurrent_classes(self):
        """Return a list of recurrent classes."""
        is_recurrent = partial(_is_recurrent_component, self._graph)
        return list(filter(is_recurrent, self.communicating_classes()))

    def transient_classes(self):
        """Return a list of transient classes."""
        is_transient = partial(_is_transient_component, self._graph)
        return list(filter(is_transient, self.communicating_classes()))

    def _absorbing_idxs(self):
        # If a row has one non-zero entry, it must be a one and the state is recurring
        mask = np.count_nonzero(np.asarray(self._P), 1) == 1
        return np.arange(self._num_states)[mask]

    def _labels_at_indices(self, indices):
        return [self.labels.inv[i] for i in indices]

    def absorbing_states(self):
        """Return a list of absorbing states, if any."""
        return self._labels_at_indices(self._absorbing_idxs())

    def transient_states(self):
        """Return a list of transient states, if any."""
        return reduce(set.union, self.transient_classes(), set())

    def recurrent_states(self):
        """Return a list of recurrent states, if any."""
        return reduce(set.union, self.recurrent_classes(), set())

    def _sub_matrix(self, indices):
        return self._P[np.ix_(indices, indices)]

    def steady_states(self):
        """Return the vector(s) of steady state(s)."""
        if self.is_reducible():
            raise NotImplementedError("Steady state distribution only implemented for irreducible chains.")

        n = self._num_states
        p = self._P.T - np.eye(self._num_states)

        sum_constraint = np.vstack((p, np.ones(n)))

        b = np.zeros(n + 1)
        b[-1] = 1

        return np.linalg.lstsq(sum_constraint, b, rcond=None)[0]

    def canonic_form(self):
        """Return the transition matrix in canonic form."""
        raise NotImplementedError

    def is_irreducible(self):
        """Check if the chain is reducible."""
        return nx.number_strongly_connected_components(self._graph) == 1

    def is_reducible(self):
        return not self.is_irreducible()

    def is_ergodic(self):
        m = (self._num_states - 1)**2 + 1
        return np.all(np.linalg.matrix_power(self._P, m) > 0)

    def mixing_time(self):
        w, _ = np.linalg.eig(self._P)
        u = np.sort(np.absolute(w))[-2]
        return -1 / np.log(u)

    def period(self, state):
        """Return the period of the given state."""
        cycles = nx.simple_cycles(self._graph)
        cycles_on_state = filter(lambda cycle: self.labels[state] in cycle, cycles)
        cycle_lengths = map(len, cycles_on_state)
        return reduce(gcd, cycle_lengths)
    # TODO: Calculate the cyclic classes (see http://math.bme.hu/~nandori/Virtual_lab/stat/markov/Periodicity.pdf)

    def conditional_distribution(self, state):
        return self._P[self.labels[state]]

    def redistribute(self, num_steps, initial_distribution=None):
        if num_steps < 1:
            raise ValueError("The number of steps must be a positive integer. Received {}".format(num_steps))
        if initial_distribution is None:
            initial_distribution = np.ones(self._num_states, dtype='float64') / self._num_states
        else:
            initial_distribution = np.asarray(initial_distribution)
            if initial_distribution.size != self._num_states:
                raise ValueError("The initial distribution must have size equal to the number of states.")
            initial_distribution /= initial_distribution.sum()

        redistribution = np.array((self._num_states, num_steps + 1))
        redistribution[0] = initial_distribution
        p_power = self._P * self._P
        for i in range(1, num_steps + 2):
            redistribution[i] = np.matmul(initial_distribution, p_power)
            p_power *= self._P
        return redistribution


