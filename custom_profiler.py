# import cProfile
# import functools
# import time

# import numpy as np
# from tabulate import tabulate


# class Profiler:

#     def __init__(self, disable=False):
#         self.disable = disable
#         self.ts: dict[str, list[float]] = {}

#     def timeit(self, f):
#         @functools.wraps(f)
#         def wrap(*args, **kwargs):
#             if self.disable:
#                 return f(*args, **kwargs)
#             if f.__name__ not in self.ts.keys():
#                 self.ts[f.__name__] = []
#             t_start = time.perf_counter()
#             result = f(*args, **kwargs)
#             t_end = time.perf_counter()
#             self.ts[f.__name__].append(t_end - t_start)
#             return result
#         return wrap

#     def print_stats(self):
#         if not self.disable:
#             print('stats')
#             for func_name, exec_times in self.ts:
#                 mean = np.mean(exec_times)
#                 std = np.std(exec_times)


# import cProfile
# import functools


# class ProfilerManager:

#     def __init__(self, disable=True) -> None:
#         self.disable = disable

#     def timeit(self, f):

#         # @functools.wraps(f)
#         # def run_func(*args, **kwargs):
#         #     if not self.profile:
#         #         result = f(*args, **kwargs)
#         #     else:
#         #         self.enable()
#         #         result = f(*args, **kwargs)
#         #         self.disable()
#         #     return result
#         # return run_func

#         # return self.runcall(f)

#         @functools.wraps(f)
#         def run_func(*args, **kwargs):
#             if not self.profile:
#                 return f(*args, **kwargs)
#             else:
#                 return self.runcall(f, *args, **kwargs)
#         return run_func

#     def print(self):
#         if self.profile:
#             # sio = io.StringIO()
#             # ps = pstats.Stats(self, stream=sio).strip_dirs().sort_stats().print_stats() # 'tottime'
#             # self.print_stats()
#             # print(sio.getvalue())
#             pass


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
