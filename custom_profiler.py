import cProfile
import functools
import io
import pstats

_PROFILER = cProfile.Profile(subcalls=False, builtins=False)

_FUNCS = []

# TODO: try to implement as class again

DEACTIVATE = False


# def timeit(func, deactivate: bool = False):
#     @functools.wraps(func)
#     def run_func(*args, **kwargs):
#         if deactivate:
#             return func(*args, **kwargs)
#         try:
#             _PROFILER.enable()
#         except ValueError:
#             return func(*args, **kwargs)
#         try:
#             return func(*args, **kwargs)
#         finally:
#             _PROFILER.disable()
#     if func.__name__ not in _FUNCS:
#         _FUNCS.append(func.__name__)
#     return run_func


def timeit(func):
    @functools.wraps(func)
    def run_func(*args, **kwargs):
        if DEACTIVATE:
            return func(*args, **kwargs)
        try:
            _PROFILER.enable()
        except ValueError:
            return func(*args, **kwargs)
        try:
            return func(*args, **kwargs)
        finally:
            _PROFILER.disable()
    if func.__name__ not in _FUNCS:
        _FUNCS.append(func.__name__)
    return run_func


def printit(clear: bool = False) -> None:
    if not DEACTIVATE:
        sio = io.StringIO()
        ps = pstats.Stats(_PROFILER, stream=sio)
        ps.strip_dirs().sort_stats(pstats.SortKey.STDNAME).print_stats('|'.join(_FUNCS))
        print(sio.getvalue())
        if clear:
            _PROFILER.clear()
