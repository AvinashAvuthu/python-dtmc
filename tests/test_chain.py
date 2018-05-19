from dtmc import dtmc
import numpy as np
import pytest


market_p = [[0.9, 0.075, 0.025], [0.15, 0.8, 0.05], [0.25, 0.25, 0.5]]
market_labels = ('bull', 'bear', 'stagnant')
market_chain = dtmc(market_p, market_labels)

weather_p = [[0.9, 0.1], [0.5, 0.5]]
weather_labels = ('sunny', 'rainy')
weather_chain = dtmc(weather_p, weather_labels)

sigman_p = [[1/2.0, 1/2.0, 0, 0],
            [1/2.0, 1/2.0, 0, 0],
            [1/3.0, 1/6.0, 1/6.0, 1/3.0],
            [0, 0, 0, 1]]
sigman_chain = dtmc(sigman_p)

periodic_p = [[0, 0, 1/2.0, 1/4.0, 1/4.0, 0, 0],
              [0, 0, 1/3.0, 0, 2/3.0, 0, 0],
              [0, 0, 0, 0, 0, 1/3.0, 2/3.0],
              [0, 0, 0, 0, 0, 1/2.0, 1/2.0],
              [0, 0, 0, 0, 0, 3/4.0, 1/4.0],
              [1/2.0, 1/2.0, 0, 0, 0, 0, 0],
              [1/4.0, 3/4.0, 0, 0, 0, 0, 0]]
periodic_chain = dtmc(periodic_p)

ravner_p = [[0, 1, 0, 0, 0, 0],
            [0.4, 0.6, 0, 0, 0, 0],
            [0.3, 0, 0.4, 0.2, 0.1, 0],
            [0, 0, 0, 0.3, 0.7, 0],
            [0, 0, 0, 0.5, 0, 0.5],
            [0, 0, 0, 0.3, 0, 0.7]]
ravner_chain = dtmc(ravner_p)

fenix_p = [[0.8, 0.2],
           [0.4, 0.6]]
fenix_chain = dtmc(fenix_p)


def frog_matrix(p, q):
    return np.asmatrix([
        [1 - p, p],
        [q, 1 - q]
    ])


# ---- Acceptance of valid transition matrices ----

valid_chains = [
    (market_p, market_labels),
    (weather_p, weather_labels),
    (sigman_p, None),
    (periodic_p, None),
    (np.eye(10), None),
    (np.eye(10).T, None),
    (ravner_p, None)
]


@pytest.mark.parametrize('matrix, labels', valid_chains)
def test_accept_valid_chain(matrix, labels):
    dtmc(matrix, labels)


# ---- Rejection of invalid transition matrices ----

bad_chains = [
    (market_p, weather_labels),
    (weather_p, market_labels),
    (-1 * np.eye(10), None),
    (np.zeros((5, 4)), None),
    (np.eye(10) + np.eye(10).T, None),
    (np.eye(3), ['a', 'b', 'a'])
]


@pytest.mark.parametrize('matrix, labels', bad_chains)
def test_reject_bad_chain(matrix, labels):
    with pytest.raises(ValueError):
        dtmc(matrix, labels)


# ---- Test absorbing states ----

def test_all_absorbing():
    mc = dtmc(np.eye(10))
    assert np.array_equal(mc.absorbing_states(), np.arange(10))


def test_all_absorbing_labelled():
    labels = [str(i) for i in range(10)]
    mc = dtmc(np.eye(10), labels)
    assert mc.absorbing_states() == labels


# ---- Test communicating classes ---

def test_multiple_communicating_classes():
    classes = map(tuple, sigman_chain.communicating_classes())
    expected_classes = map(tuple, [{0, 1}, {2}, {3}])
    assert sorted(classes) == sorted(expected_classes)


# ---- Test irreducibility ----

reducible_chains = [
    (sigman_p, None),
    (np.eye(10), None)
]
irreducible_chains = [
    (market_p, market_labels),
    (weather_p, weather_labels)
]


@pytest.mark.parametrize('matrix, labels', reducible_chains)
def test_reducible_chain(matrix, labels):
    assert dtmc(matrix, labels).is_reducible()


@pytest.mark.parametrize('matrix, labels', irreducible_chains)
def test_irreducible_chain(matrix, labels):
    assert dtmc(matrix, labels).is_irreducible()


# ---- Test periodicity ----
def test_aperiodic_chain():
    assert weather_chain.period('sunny') == 1
    assert market_chain.period('bull') == 1


def test_periodic_chain():
    assert periodic_chain.period(0) == 3


# ---- Test transience and recurrence ----

def test_transient_classes():
    mc = dtmc(ravner_p)
    t_classes = mc.transient_classes()
    assert t_classes == [{2}]


def test_recurrent_classes():
    mc = dtmc(ravner_p)
    r_classes = mc.recurrent_classes()
    assert r_classes == [{0, 1}, {3, 4, 5}]


def test_transient_states():
    mc = dtmc(ravner_p)
    t_states = mc.transient_states()
    assert t_states == {2}


def test_recurrent_states():
    mc = dtmc(ravner_p)
    r_states = mc.recurrent_states()
    assert r_states == {0, 1, 3, 4, 5}


# ---- Test steady state distribution ----

def test_steady_state():
    mc = dtmc(fenix_p)
    assert np.allclose(mc.steady_states(), [2/3.0, 1/3.0])


# ---- Test redistribution ----

def test_market_redistribution():
    init = [0, 1, 0]  # Start in a bear market
    redistribution = market_chain.redistribute(2, init)
    assert np.allclose(redistribution[-1], [0.3575, 0.56825, 0.07425])


# ---- Test stationary distribution ----
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import builds, composite, integers, floats
from hypothesis import given


@composite
def right_stochastic_matrices(draw):
    def normalize(matrix):
        matrix = np.abs(matrix)
        row_sums = matrix.sum(axis=1, keepdims=True)
        return np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)
    size = draw(integers(2, 25))
    return normalize(draw(arrays(np.float64, (size, size), floats(1, 100, False, False))))


@given(right_stochastic_matrices())
def test_accept_right_stochastic_matrix(mat):
    dtmc(mat)


@given(floats(0.1, .9, False, False), floats(0.1, .9, False, False))
def test_frog_stationary_distribution(p, q):
    mc = dtmc(frog_matrix(p, q))
    assert np.allclose(mc.steady_states(), [q/(p+q), (p/(p+q))])

