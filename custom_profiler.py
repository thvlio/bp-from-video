import cProfile
import functools
import io
import pstats

from config import Config as c


class CustomProfiler(cProfile.Profile):

    # TODO: change to Profiler and file to profiler.py

    def __init__(self, enabled: bool = c.PROFILER_ENABLED):
        self.enabled = enabled
        self.funcs = []
        super().__init__(subcalls=False, builtins=False)

    def timeit(self, func):
        @functools.wraps(func)
        def run_func(*args, **kwargs):
            if not self.enabled:
                return func(*args, **kwargs)
            try:
                self.enable()
            except ValueError:
                return func(*args, **kwargs)
            try:
                return func(*args, **kwargs)
            finally:
                self.disable()
        if func.__name__ not in self.funcs:
            self.funcs.append(func.__name__)
        return run_func

    def printit(self, clear_info: bool = False) -> None:
        if self.enabled:
            sio = io.StringIO()
            ps = pstats.Stats(self, stream=sio)
            ps.strip_dirs().sort_stats(pstats.SortKey.STDNAME).print_stats('|'.join(self.funcs))
            print(sio.getvalue())
            if clear_info:
                self.clear()


profiler = CustomProfiler()
