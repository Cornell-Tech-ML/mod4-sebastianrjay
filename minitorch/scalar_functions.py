from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the function to a set of variables, returning the output as a new Scalar instance."""
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for addition"""
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Backward pass for addition"""
        return d_output, d_output


class Sub(ScalarFunction):
    """Subtraction function $f(x, y) = x - y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for subtraction"""
        ctx.save_for_backward(a, b)
        return operators.sub(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Backward pass for subtraction"""
        return d_output, -d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for log"""
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for log"""
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.


# TODO: Implement for Task 1.2.


class EQ(ScalarFunction):
    """Equality check function $f(x, y) = 1.0 if x == y, 0.0 o.w."""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for equality check"""
        return 1.0 if a == b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for equality check"""
        return 0.0, 0.0


class GT(ScalarFunction):
    """Less than check function $f(x, y) = 1.0 if x > y, 0.0 o.w."""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for less than check"""
        return 1.0 if a > b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for less than check"""
        return 0.0, 0.0


class LT(ScalarFunction):
    """Less than check function $f(x, y) = 1.0 if x < y, 0.0 o.w."""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for less than check"""
        return 1.0 if a < b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for less than check"""
        return 0.0, 0.0


class Exp(ScalarFunction):
    """Calculates e to the power of x, i.e. the natural exponential function."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass of e to the power of x"""
        out = operators.exp(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass of e to the power of x"""
        out: float = ctx.saved_values[0]
        return d_output * out


class Inv(ScalarFunction):
    """Implement the inverse function."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass of the inverse function"""
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass of the inverse function"""
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Mul(ScalarFunction):
    """Multiplication function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass of the multiplication function"""
        ctx.save_for_backward(a, b)
        c = a * b
        return c

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Backward pass of the multiplication function"""
        (a, b) = ctx.saved_values
        return b * d_output, a * d_output


class Neg(ScalarFunction):
    """Return -1 times a"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass of the negation function"""
        return -a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass of the negation function"""
        return -d_output


class ReLU(ScalarFunction):
    """Implements the relu function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass of the ReLU function"""
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass of the ReLU"""
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Sigmoid(ScalarFunction):
    """Implement the sigmoid function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass of the sigmoid function"""
        out = operators.sigmoid(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass of the sigmoid function"""
        sigma: float = ctx.saved_values[0]
        return sigma * (1.0 - sigma) * d_output
