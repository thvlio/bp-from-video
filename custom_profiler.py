import cProfile
import functools
import io
import pstats


class Profiler:

    def __init__(self, deactivate: bool = False):
        self._profiler = cProfile.Profile(subcalls=False, builtins=False)
        self._funcs = []
        self.deactivate = deactivate

    def timeit(self, func):
        @functools.wraps(func)
        def run_func(*args, **kwargs):
            if self.deactivate:
                return func(*args, **kwargs)
            try:
                self._profiler.enable()
            except ValueError:
                return func(*args, **kwargs)
            try:
                return func(*args, **kwargs)
            finally:
                self._profiler.disable()
        if func.__name__ not in self._funcs:
            self._funcs.append(func.__name__)
        return run_func

    def printit(self, clear: bool = False) -> None:
        if not self.deactivate:
            sio = io.StringIO()
            ps = pstats.Stats(self._profiler, stream=sio)
            ps.strip_dirs().sort_stats(pstats.SortKey.STDNAME).print_stats('|'.join(self._funcs))
            print(sio.getvalue())
            if clear:
                self._profiler.clear()


profiler = Profiler()
