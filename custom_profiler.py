import cProfile
import functools
import io
import pstats

from config import Config as c


class Profiler:

    def __init__(self, activate: bool = c.PROFILER_ACTIVATE):
        self.activate = activate
        self.profiler = cProfile.Profile(subcalls=False, builtins=False)
        self.funcs = []

    def timeit(self, func):
        @functools.wraps(func)
        def run_func(*args, **kwargs):
            if not self.activate:
                return func(*args, **kwargs)
            try:
                self.profiler.enable()
            except ValueError:
                return func(*args, **kwargs)
            try:
                return func(*args, **kwargs)
            finally:
                self.profiler.disable()
        if func.__name__ not in self.funcs:
            self.funcs.append(func.__name__)
        return run_func

    def printit(self, clear_info: bool = False) -> None:
        if self.activate:
            sio = io.StringIO()
            ps = pstats.Stats(self.profiler, stream=sio)
            ps.strip_dirs().sort_stats(pstats.SortKey.STDNAME).print_stats('|'.join(self.funcs))
            print(sio.getvalue())
            if clear_info:
                self.profiler.clear()


profiler = Profiler()
