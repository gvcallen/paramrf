"""
Base optimization functions and classes.
"""
import warnings
from typing import Any, Callable, TypeAlias
import abc

from jaxtyping import Array, PyTree, Scalar
import equinox as eqx
import jax
from pmrf._solver_view import SolverView
from pmrf.objectives.problems import AbstractProblem
from pmrf.parameters import tree_param_paths


class MinimizeResult(eqx.Module):
    """The core mathematical payload of a minimization run."""
    #: The optimal arrays (y_opt)
    y: PyTree
    
    #: Whether the algorithm successfully converged
    success: bool = True
    
    #: Any solver metrics.
    metrics: Any = None

    #: The optimal hyperparameters, for a least-squares minimiser given any, else None.
    hyper: PyTree = None


class AbstractUnconstrainedMinimizer(eqx.Module):
    """
    Abstract interface for unconstrained minimization algorithms.

    It does not honour bounds, so it moves through raw space, where each parameter's
    bijector keeps it inside its bounds.
    """

    @property
    def honours_bounds(self) -> bool:
        """Whether the minimiser keeps its values inside the bounds it is given: never."""
        return False

    @abc.abstractmethod
    def run(
        self,
        fn: Callable[[PyTree, Any], Scalar],
        y0: PyTree,
        args: Any,
        max_iter: int = 1024,
        **kwargs
    ) -> MinimizeResult:
        """
        Execute the minimization algorithm.

        Parameters
        ----------
        fn : callable
            The objective function to minimize.
        y0 : PyTree
            The initial parameter guess.
        args : Any
            Args to pass to `fn`.            
        max_iter: int = 1024
            The maximum number of iterations to take.
        **kwargs
            Runtime arguments forward to the solver backend.

        Returns
        -------
        results
            An instance of :class:`pmrf.optimize.MinimizeResult`.
        """
        raise NotImplementedError
    

class AbstractBoundedMinimizer(eqx.Module):
    """
    Abstract interface for bounded minimization algorithms.

    A minimiser that honours bounds (:attr:`honours_bounds`) searches box space, the
    box of the parameters' bounds, and is given its edges as `bounds`. One that does
    not, such as a backend configured with an unbounded method, moves through raw
    space instead and is given no `bounds`.
    """

    @property
    def honours_bounds(self) -> bool:
        """Whether the minimiser keeps its values inside the bounds it is given.

        True by default. A subclass whose backend can ignore bounds overrides it.
        """
        return True

    @abc.abstractmethod
    def run(
        self,
        fn: Callable[[PyTree, Any], Scalar],
        y0: PyTree,
        args: Any,
        bounds: tuple[PyTree, PyTree] | None = None,
        max_iter: int = 1024,
        **kwargs
    ) -> MinimizeResult:
        """
        Execute the minimization algorithm.

        Parameters
        ----------
        fn : callable
            The objective function to minimize.
        y0 : PyTree
            The initial parameter guess.
        args : Any
            Args to pass to `fn`.
        bounds : tuple[PyTree, PyTree], optional
            The lower and upper edges of the box to search, shaped like `y0`, or None
            if the minimiser does not honour bounds. Edges may be infinite.
        max_iter: int = 1024
            The maximum number of iterations to take.
        **kwargs
            Runtime arguments forward to the solver backend.

        Returns
        -------
        results
            An instance of :class:`pmrf.optimize.MinimizeResult`.
        """
        raise NotImplementedError
    

class AbstractLeastSquaresMinimizer(eqx.Module):
    """
    Abstract interface for bounded least-squares minimisation algorithms.

    A least-squares minimiser minimises half the squared norm of a residual vector
    rather than a scalar objective (ADR-0009). It honours bounds, so it searches box
    space and is given the box's edges as `bounds`.

    It may also be given hyperparameters: values the residual depends on but that are
    not least-squares variables, with a scalar objective of them. It then alternates a
    least-squares solve over `y` with the hyperparameters held and a step over the
    hyperparameters with `y` held. It knows nothing of what the hyperparameters are.
    """

    @property
    def honours_bounds(self) -> bool:
        """Whether the minimiser keeps its values inside the bounds it is given: always."""
        return True

    @abc.abstractmethod
    def run(
        self,
        residual_fn: Callable[[PyTree, PyTree, Any], Array],
        y0: PyTree,
        args: Any,
        bounds: tuple[PyTree, PyTree] | None = None,
        max_iter: int = 1024,
        *,
        hyper_fn: Callable[[PyTree, PyTree, Any], Scalar] | None = None,
        hyper0: PyTree = None,
        hyper_bounds: tuple[PyTree, PyTree] | None = None,
        **kwargs
    ) -> MinimizeResult:
        """
        Execute the minimization algorithm.

        Parameters
        ----------
        residual_fn : callable
            The residual function `residual_fn(y, hyper, args)`, returning a real 1-D
            vector. Half its squared norm is minimised over `y`. `hyper` is None when
            there are no hyperparameters.
        y0 : PyTree
            The initial parameter guess.
        args : Any
            Args to pass to `residual_fn` and `hyper_fn`.
        bounds : tuple[PyTree, PyTree], optional
            The lower and upper edges of the box to search, shaped like `y0`. Edges may
            be infinite.
        max_iter: int = 1024
            The maximum number of iterations of each solve.
        hyper_fn : callable, optional
            The full scalar objective `hyper_fn(hyper, y, args)`, minimised over the
            hyperparameters, or None. Required with `hyper0`; without it, only the
            objective reported is affected.
        hyper0 : PyTree, optional
            The initial hyperparameters, or None if there are none.
        hyper_bounds : tuple[PyTree, PyTree], optional
            The lower and upper edges of the hyperparameters' box, shaped like `hyper0`.
        **kwargs
            Runtime arguments forward to the solver backend.

        Returns
        -------
        results
            An instance of :class:`pmrf.optimize.MinimizeResult`, with the
            hyperparameters in `hyper`.
        """
        raise NotImplementedError


#: A type-hint for a minimizer in :mod:`pmrf.optimize`. One of
#: :class:`pmrf.optimize.AbstractUnconstrainedMinimizer`,
#: :class:`pmrf.optimize.AbstractBoundedMinimizer` or
#: :class:`pmrf.optimize.AbstractLeastSquaresMinimizer`.
AbstractMinimizer: TypeAlias = AbstractUnconstrainedMinimizer | AbstractBoundedMinimizer | AbstractLeastSquaresMinimizer
    

def is_minimizer(x):
    """
    Returns True if x is an instance of `pmrf.optimize.AbstractUnconstrainedMinimizer`,
    `pmrf.optimize.AbstractBoundedMinimizer` or `pmrf.optimize.AbstractLeastSquaresMinimizer`.
    """
    return isinstance(x, AbstractMinimizer)


def is_optimizer(x):
    """Returns True if `pmrf.optimize.is_minimizer` returns True."""
    return is_minimizer(x)


def run_minimizer(
    fn: Callable[[PyTree, Any], Scalar],
    model: PyTree,
    solver: AbstractMinimizer,
    args: Any = None,
    max_iter: int = 1024,
    residual_fn: Callable[[PyTree, Any], Array] | None = None,
    **kwargs
) -> tuple[PyTree, MinimizeResult]:
    """
    Optimizes the free parameters of a model, or any collection of models and parameters.

    The solver can be any solver of type `pmrf.optimize.AbstractMinimizer`. It receives
    the free parameters as a name-keyed dict; a solver that needs a 1-D vector flattens
    it itself. The space depends on whether the solver honours bounds
    (``solver.honours_bounds``):

    - **Box space**, if it does. Each parameter is searched over the box of its bounds,
      not its prior: the unit box between two finite bounds, declared space otherwise.
      The solver is given the box's edges as `bounds`. A closed edge is the bound
      itself, so a parameter can start on it, reach it and stay on it; each finite open
      edge is inset by a dtype-aware relative distance that preserves valid interior
      starts.
    - **Raw space**, if it does not. Constraints are enforced by each parameter's
      bijector. A parameter starting exactly on a closed bound has an infinite raw
      value, so the solver starts it just inside: by 1e-6 of the width between two
      finite bounds, or 1e-6 in declared units otherwise.

    A least-squares minimiser (:class:`AbstractLeastSquaresMinimizer`) searches box space
    too, but is handed `residual_fn` instead of `fn`. When `model` is a problem
    (:class:`pmrf.objectives.problems.AbstractProblem`), its free parameters are split
    (ADR-0009): the **model parameters** are those of ``model.model``, and the
    **hyperparameters** are the rest, those the terms own. The minimiser is handed the
    residual as a function of the model parameters given the hyperparameters, and `fn`
    as the full objective of the hyperparameters given the model parameters. For any
    other tree every free parameter is a model parameter. The residual is traced once at
    the start, so a term without a whitened residual raises before the solver runs.

    The result is written back with :func:`pmrf.update`, so fixed parameters, names,
    scales and priors are unchanged.

    Parameters
    ----------
    fn : callable
        The objective function taking `(unwrapped_model, args)`.
    model : PyTree
        The initial model.
    solver : AbstractMinimizer
        The instantiated optimizer to run.
    args : Any, optional
        Args to pass to `fn`. Defaults to None.
    max_iter : int, optional
        Maximum number of iterations. Defaults to 1024.
    residual_fn : callable, optional
        The residual function taking `(unwrapped_model, args)`, half of whose squared
        norm is `fn` up to terms constant in the model parameters. Required by a
        least-squares minimiser, ignored by any other.
    **kwargs
        Runtime arguments forwarded to the solver backend.

    Returns
    -------
    tuple
        A tuple of `(best_model, minimize_results)`.

    Raises
    ------
    ValueError
        If `model` has no free parameters, or one starts at NaN, or a least-squares
        minimiser is given no `residual_fn`.
    NotImplementedError
        If a least-squares minimiser is used with a term that has no whitened residual.
    FloatingPointError
        If a SciPy minimizer evaluates a nonfinite objective, requested gradient or
        attempted optimizer vector.
    """
    view = SolverView(model, 'optimize')
    # A start at NaN cannot move in either space. A start on a closed bound is already
    # nudged inside it in raw space.
    view.check_finite()
    if solver.honours_bounds:
        space = 'box'
        y0, kwargs['bounds'] = view.box()
    else:
        space, y0 = 'raw', view.y0
    if isinstance(solver, AbstractLeastSquaresMinimizer):
        result = _run_least_squares(fn, residual_fn, model, view, solver, y0, args, max_iter, **kwargs)
        values = {**result.y, **(result.hyper or {})}
    else:
        result = solver.run(fn=view.objective(fn, space), y0=y0, args=args, max_iter=max_iter, **kwargs)
        values = result.y
    opt_model = view.updated(values, space)

    if not result.success:
        warnings.warn("Optimization failed to converge. Try increasing the maximum number of iterations or loosening the solver tolerances.")

    return opt_model, result


def _model_parameter_names(tree: PyTree) -> set[str]:
    """Returns the names of the free parameters of a problem's model, or of every free
    parameter if `tree` is not a problem."""
    paths = tree_param_paths(tree, free_only=True)
    if not isinstance(tree, AbstractProblem):
        return set(paths)
    model = tree.model
    path_leaves, _ = jax.tree_util.tree_flatten_with_path(tree, is_leaf=lambda x: x is model)
    prefixes = [path for path, leaf in path_leaves if leaf is model]
    if not prefixes:
        raise ValueError("The problem's model could not be located in the problem.")
    prefix = prefixes[0]
    return {name for name, (path, _) in paths.items() if tuple(path[:len(prefix)]) == tuple(prefix)}


def _run_least_squares(
    fn, residual_fn, model, view, solver, y0, args, max_iter, *, bounds, **kwargs
) -> MinimizeResult:
    """Splits the free parameters of `model` and runs a least-squares minimiser on them,
    in box space."""
    if residual_fn is None:
        raise ValueError(
            f"{type(solver).__name__} is a least-squares minimiser and needs a residual "
            "function, but none was given."
        )
    model_names = _model_parameter_names(model)
    lower, upper = bounds

    def split(values):
        return (
            {name: v for name, v in values.items() if name in model_names},
            {name: v for name, v in values.items() if name not in model_names},
        )

    y_model, y_hyper = split(y0)
    if not y_model:
        raise ValueError(
            "Nothing to optimize by least squares: the problem's model has no free "
            "parameters, only hyperparameters."
        )
    (lower_model, lower_hyper), (upper_model, upper_hyper) = split(lower), split(upper)
    box_residual_fn = view.objective(residual_fn, 'box')
    box_fn = view.objective(fn, 'box')

    def model_residual_fn(y, hyper, a):
        return box_residual_fn({**y, **(hyper or {})}, a)

    def hyper_fn(hyper, y, a):
        return box_fn({**y, **(hyper or {})}, a)

    hyper0 = y_hyper or None
    # Tracing raises for a term without a whitened residual, before the solver runs.
    jax.eval_shape(model_residual_fn, y_model, hyper0, args)
    return solver.run(
        residual_fn=model_residual_fn, y0=y_model, args=args,
        bounds=(lower_model, upper_model), max_iter=max_iter,
        hyper_fn=hyper_fn, hyper0=hyper0,
        hyper_bounds=(lower_hyper, upper_hyper) if hyper0 is not None else None,
        **kwargs,
    )
