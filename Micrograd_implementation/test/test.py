"""
Тесты для проверки:
- обратного распространения в `micrograd.engine.Value`
- создания и работы `micrograd.nn.MLP`
"""

import math
import random

from micrograd.engine import Value
from micrograd.nn import MLP


def test_backward_polynomial_chain():
    # f(a,b) = (a*b + a + b)^2
    a = Value(2.0)
    b = Value(3.0)

    g = a * b + a + b  # g = 11
    f = g ** 2          # f = 121
    f.backward()

    # Аналитические производные:
    # df/dg = 2*g = 22
    # dg/da = b + 1 = 4  => df/da = 22 * 4 = 88
    # dg/db = a + 1 = 3  => df/db = 22 * 3 = 66
    assert abs(f.data - 121.0) < 1e-9
    assert abs(a.grad - 88.0) < 1e-6
    assert abs(b.grad - 66.0) < 1e-6


def test_backward_tanh_chain_rule():
    # f(a,b) = tanh(a + b)
    a = Value(1.0)
    b = Value(2.0)
    s = a + b
    f = s.tanh()
    f.backward()

    t = math.tanh(3.0)
    df_ds = 1 - t * t
    # ds/da = 1, ds/db = 1
    assert abs(f.data - t) < 1e-9
    assert abs(a.grad - df_ds) < 1e-6
    assert abs(b.grad - df_ds) < 1e-6


def test_mlp_creation_and_backward_flow():
    # Фиксируем сид для детерминированной инициализации весов
    random.seed(42)

    # MLP: 3 -> 4 -> 4 -> 1
    mlp = MLP(3, [4, 4, 1])

    # Кол-во параметров: 4*(3+1) + 4*(4+1) + 1*(4+1) = 41
    params = mlp.parameters()
    assert len(params) == 41

    # Прямой проход: вход — список Value из того же модуля `engine`
    x = [Value(1.0), Value(2.0), Value(-1.0)]
    y_pred = mlp(x)

    # Последний слой размером 1 => на выходе один Value
    assert isinstance(y_pred, Value)

    # Небольшая целевая функция: квадратичная ошибка к нулю
    loss = (y_pred - 0.0) ** 2
    mlp.zero_grad()
    loss.backward()

    # Градиенты должны дойти хотя бы до части параметров
    total_grad_sq = sum((p.grad ** 2 for p in params))
    assert total_grad_sq > 0.0

    # Проверим, что сброс градиентов работает
    mlp.zero_grad()
    assert all(p.grad == 0.0 for p in params)


def test_reverse_ops_and_sum_behavior():
    v = Value(2.0)
    # Reverse subtraction and addition
    assert (1 - v).data == -1.0
    assert (1 + v).data == 3.0
    # Sum over Values works both with and without a start value
    s1 = sum([Value(1.0), Value(2.0)])
    s2 = sum([Value(1.0), Value(2.0)], Value(0.0))
    assert isinstance(s1, Value) and isinstance(s2, Value)
    assert s1.data == 3.0 and s2.data == 3.0


def test_mlp_last_layer_linear_flag():
    random.seed(0)
    mlp = MLP(3, [4, 1])
    # Последний слой должен быть линейным
    last_layer = mlp.layers[-1]
    assert all(not n.nonlin for n in last_layer.layer)
