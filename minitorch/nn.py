from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    out_height = height // kh
    out_width = width // kw

    # split original width into tile size kw
    input = input.contiguous().view(batch, channel, height, out_width, kw)
    # reorder dims and split original height into tile size kh
    input = (
        input.permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(batch, channel, out_height, out_width, kh * kw)
    )

    return input, out_height, out_width


# TODO: Implement for Task 4.3.
# - avgpool2d: Tiled average pooling 2D
def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Performs average pooling over 2D using tiling
    Args:
        input: 4D tensor (batch, channel, width, height)
        kernel: height, width to pool
    Returns:
        Pooled tensor of the same batch and channel size
    """
    batch, channel, height, width = input.shape
    input, out_height, out_width = tile(input, kernel)

    output = input.mean(4).view(batch, channel, out_height, out_width)
    return output


fastops_max_reduce = FastOps.reduce(
    operators.max, -1e9
)  # start a little negative to account for zero


# - argmax: Compute the argmax as a 1-hot tensor
def argmax(input: Tensor, dim: int) -> Tensor:
    """Finds the argmax and returns a one hot tensor to represent it (using fastops numba)

    Args:
    ----
        input: tensor
        dim: dimension to perform argmax over

    Returns:
    -------
        Tensor: one hot tensor that indicates the argmax

    """
    return fastops_max_reduce(input, dim) == input


# - Max: New Function for max operator
class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Reduction of tensor using max along given dimension"""
        ctx.save_for_backward(input, dim)
        return fastops_max_reduce(input, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad: Tensor) -> Tuple[Tensor, float]:
        """Backward for max, derivate of max is argmax"""
        (input, dim) = ctx.saved_values
        return argmax(input, int(dim.item())) * grad, 0


# - max: Apply max reduction
def max(input: Tensor, dim: int) -> Tensor:
    """Reduction of input tensor using max along given dimension"""
    return Max.apply(input, input._ensure_tensor(dim))


# - softmax: Compute the softmax as a tensor
def softmax(input: Tensor, dim: int) -> Tensor:
    r"""Compute softmax of input tensor $\frac{e^{x_i}}{\sum_i e^{x_i}}$"""
    exponented = input.exp()
    return exponented / exponented.sum(dim)


# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
def logsoftmax(input: Tensor, dim: int) -> Tensor:
    r"""Compute the log of softmax $x_i - \log\sum_i e^{e_i}$"""
    return input - input.exp().sum(dim).log()


# - maxpool2d: Tiled max pooling 2D
def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Max pooling over tensor in 2D
    Args:
        input: 4D tensor  x height x width
        kernel: size to pool over height x width
    Returns:
        Tensor: pooled 4D tensor
    """
    batch, channel, height, width = input.shape
    input, out_height, out_width = tile(input, kernel)
    return Max.apply(input, tensor(4)).view(batch, channel, out_height, out_width)


# - dropout: Dropout positions based on random noise, include an argument to turn off
def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Dropout random values in NN
    Args:
        input: input Tensor
        rate: probability of each value dropping
    Returns:
        Tensor: input tensor with random values dropped
    """
    if ignore:
        return input

    random = rand(input.shape)
    return input * (random > rate)
