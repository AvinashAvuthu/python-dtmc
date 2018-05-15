# -*- coding: utf-8 -*-

"""Main module."""

from functools import reduce, partial
try:
    from math import gcd
except ImportError:
    from fractions import gcd
import networkx as nx
import numpy as np


def _reachable_from_component(graph, component):
    reachable = partial(nx.descendants, graph)
    return reduce(set.union, map(reachable, component), set())


def _is_recurrent_component(graph, component):
    return _reachable_from_component(graph, component) == set(component)


def _is_transient_component(graph, component):
    return _reachable_from_component(graph, component) - set(component) != set()


class DiscreteTimeMarkovChain(object):
    def __init__(self, transition_matrix, labels=None):
        self._P = np.asmatrix(transition_matrix, 'float64')
        self._num_states = self._P.shape[0]

        # Check that we got a square matrix
        if self._P.shape[0] != self._P.shape[1]:
            raise ValueError("The transition matrix must be square. Got shape {}.".format(self._P.shape))

        # Check that we got a right stochastic matrix
        if not np.all(self._P >= 0):
            raise ValueError("The transition matrix can only contain non-negative values.")
        if not np.allclose(np.sum(self._P, 1), np.ones(self._num_states)):
            raise ValueError("The transition matrix must have rows that sum to (almost) 1")

        if labels is None:
            self.labels = np.arange(self._num_states)
        else:
            if len(labels) != self._num_states:
                raise ValueError("The number of labels given must equal the number of items in the transition matrix." +
                                 "Got {} labels for a matrix with {} items.".format(len(labels), self._num_states))
            if len(set(labels)) != len(labels):
                raise ValueError("The labels must be unique.")
            self.labels = np.asarray(labels)

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
        return np.count_nonzero(np.asarray(self._P), 1) == 1

    def absorbing_states(self):
        """Return a list of absorbing states, if any."""
        # If a row has one non-zero entry, it must be a one and the state is recurring
        return self.labels[self._absorbing_idxs()]

    def transient_states(self):
        """Return a list of transient states, if any."""
        return reduce(set.union, self.transient_classes(), set())

    def recurrent_states(self):
        """Return a list of recurrent states, if any."""
        return reduce(set.union, self.recurrent_classes(), set())

    def steady_states(self):
        """Return the vector(s) of steady state(s)."""
        if self.is_reducible():
            raise NotImplementedError("Steady state distribution only implemented for irreducible chains.")

        n = self._num_states
        p = self._P.T - np.eye(self._num_states)

        sum_constraint = np.vstack((p, np.ones(n)))

        b = np.zeros(n + 1)
        b[-1] = 1

        return np.linalg.lstsq(sum_constraint, b)[0]

    def canonic_form(self):
        """Return the transition matrix in canonic form."""
        raise NotImplementedError

    def is_irreducible(self):
        """Check if the chain is reducible."""
        return nx.number_strongly_connected_components(self._graph) == 1

    def is_reducible(self):
        return not self.is_irreducible()

    def period(self, state):
        """Return the period of the given state."""
        cycles = nx.simple_cycles(self._graph)
        state_idx = np.where(self.labels == state)[0][0]  # TODO: there is likely a better way
        cycles_on_state = filter(lambda cycle: state_idx in cycle, cycles)
        cycle_lengths = map(len, cycles_on_state)
        return reduce(gcd, cycle_lengths)
    # TODO: Calculate the cyclic classes (see http://math.bme.hu/~nandori/Virtual_lab/stat/markov/Periodicity.pdf)
