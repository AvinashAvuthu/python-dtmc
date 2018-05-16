#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `dtmc` package."""

import pytest


from dtmc import DiscreteTimeMarkovChain

import numpy as np

market_p = [[0.9, 0.075, 0.025], [0.15, 0.8, 0.05], [0.25, 0.25, 0.5]]
market_labels = ('bull', 'bear', 'stagnant')

weather_p = [[0.9, 0.1], [0.5, 0.5]]
weather_labels = ('sunny', 'rainy')

sigman_p = [[1/2.0, 1/2.0, 0, 0],
            [1/2.0, 1/2.0, 0, 0],
            [1/3.0, 1/6.0, 1/6.0, 1/3.0],
            [0, 0, 0, 1]]

periodic_p = [[0, 0, 1/2.0, 1/4.0, 1/4.0, 0, 0],
              [0, 0, 1/3.0, 0, 2/3.0, 0, 0],
              [0, 0, 0, 0, 0, 1/3.0, 2/3.0],
              [0, 0, 0, 0, 0, 1/2.0, 1/2.0],
              [0, 0, 0, 0, 0, 3/4.0, 1/4.0],
              [1/2.0, 1/2.0, 0, 0, 0, 0, 0],
              [1/4.0, 3/4.0, 0, 0, 0, 0, 0]]


# --- Acceptance of valid transition matrices ----

def test_accept_identity_matrix():
    """Verify the identity matrix is a valid transition matrix."""
    DiscreteTimeMarkovChain(np.eye(10))
    DiscreteTimeMarkovChain(np.eye(10).T)


def test_accept_market_example():
    DiscreteTimeMarkovChain(market_p, market_labels)


def test_accept_weather_example():
    DiscreteTimeMarkovChain(weather_p, weather_labels)


# ---- Rejection of invalid transition matrices ----

def test_reject_wrong_num_labels():
    with pytest.raises(ValueError):
        DiscreteTimeMarkovChain(market_p, weather_labels)
    with pytest.raises(ValueError):
        DiscreteTimeMarkovChain(weather_p, market_labels)


def test_reject_negative_values():
    with pytest.raises(ValueError):
        DiscreteTimeMarkovChain(-1 * np.eye(10))


def test_reject_non_square_matrix():
    with pytest.raises(ValueError):
        DiscreteTimeMarkovChain(np.zeros((5, 4)))


def test_reject_non_stochastic():
    with pytest.raises(ValueError):
        DiscreteTimeMarkovChain(np.eye(10) + np.eye(10).T)


def test_reject_duplicate_labels():
    with pytest.raises(ValueError):
        labels = ('a', 'b', 'a')
        m = np.eye(3)
        DiscreteTimeMarkovChain(m, labels)


# ---- Test absorbing states ----

def test_all_absorbing():
    mc = DiscreteTimeMarkovChain(np.eye(10))
    assert np.array_equal(mc.absorbing_states(), np.arange(10))


def test_all_absorbing_labelled():
    labels = [str(i) for i in range(10)]
    mc = DiscreteTimeMarkovChain(np.eye(10), labels)
    assert mc.absorbing_states() == labels


# ---- Test communicating classes ---

def test_multiple_communicating_classes():
    mc = DiscreteTimeMarkovChain(sigman_p)
    classes = map(tuple, mc.communicating_classes())
    expected_classes = map(tuple, [{0, 1}, {2}, {3}])
    assert sorted(classes) == sorted(expected_classes)


# ---- Test irreducibility ----

def test_reducible_chain():
    assert DiscreteTimeMarkovChain(sigman_p).is_reducible()
    assert DiscreteTimeMarkovChain(np.eye(10)).is_reducible()


def test_irreducible_chain():
    assert DiscreteTimeMarkovChain(market_p, market_labels).is_irreducible()
    assert DiscreteTimeMarkovChain(weather_p, weather_labels).is_irreducible()


# ---- Test periodicity ----

def test_aperiodic_chain():
    assert DiscreteTimeMarkovChain(weather_p, weather_labels).period('sunny') == 1
    assert DiscreteTimeMarkovChain(market_p, market_labels).period('bull') == 1


def test_periodic_chain():
    assert DiscreteTimeMarkovChain(periodic_p).period(0) == 3


# ---- Test transience and recurrence ----

ravner_p = [[0, 1, 0, 0, 0, 0],
            [0.4, 0.6, 0, 0, 0, 0],
            [0.3, 0, 0.4, 0.2, 0.1, 0],
            [0, 0, 0, 0.3, 0.7, 0],
            [0, 0, 0, 0.5, 0, 0.5],
            [0, 0, 0, 0.3, 0, 0.7]]


def test_transient_classes():
    mc = DiscreteTimeMarkovChain(ravner_p)
    t_classes = mc.transient_classes()
    assert t_classes == [{2}]


def test_recurrent_classes():
    mc = DiscreteTimeMarkovChain(ravner_p)
    r_classes = mc.recurrent_classes()
    assert r_classes == [{0, 1}, {3, 4, 5}]


def test_transient_states():
    mc = DiscreteTimeMarkovChain(ravner_p)
    t_states = mc.transient_states()
    assert t_states == {2}


def test_recurrent_states():
    mc = DiscreteTimeMarkovChain(ravner_p)
    r_states = mc.recurrent_states()
    assert r_states == {0, 1, 3, 4, 5}


# ---- Test steady state distribution ----

fenix_p = [[0.8, 0.2],
           [0.4, 0.6]]


def test_steady_state():
    mc = DiscreteTimeMarkovChain(fenix_p)
    assert np.allclose(mc.steady_states(), [2/3.0, 1/3.0])
