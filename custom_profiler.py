import cProfile
import functools
import io
import pstats


PROFILER = cProfile.Profile(subcalls=False, builtins=False)

FUNCS = []


def timeit(f, deactivate=False):
    @functools.wraps(f)
    def run_func(*args, **kwargs):
        if deactivate:
            return f(*args, **kwargs)
        try:
            PROFILER.enable()
        except ValueError:
            return f(*args, **kwargs)
        try:
            return f(*args, **kwargs)
        finally:
            PROFILER.disable()
    if f.__name__ not in FUNCS:
        FUNCS.append(f.__name__)
    return run_func


def printit(clear=False, deactivate=False):
    if not deactivate:
        sio = io.StringIO()
        ps = pstats.Stats(PROFILER, stream=sio)
        ps.strip_dirs().sort_stats(pstats.SortKey.STDNAME).print_stats('|'.join(FUNCS))
        print(sio.getvalue())
        if clear:
            PROFILER.clear()
