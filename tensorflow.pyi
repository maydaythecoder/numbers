from typing import Any, Callable

class _Keras:
    datasets: Any
    utils: Any
    models: Any
    layers: Any

class _NN:
    leaky_relu: Callable[..., Any]
    softmax: Callable[..., Any]

keras: _Keras
nn: _NN