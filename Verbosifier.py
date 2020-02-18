import time

VERBOSIFY = False
CALLS = {}

def enableVerbosity():
    global VERBOSIFY
    VERBOSIFY = True

def verbose(frequency = 1):
    def decorator(func):
        name = func.__name__
        CALLS[name] = [0, 0]
        accumulator = CALLS[name]
        def f(*args, **kwargs):
            s = time.time()
            ret = func(*args, **kwargs)
            e = time.time()
            accumulator[0] += 1
            accumulator[-1] += e - s
            if VERBOSIFY and accumulator[0] % frequency == 0:
                print(f'{name} | {accumulator[-1] / accumulator[0]}')
            return ret
        return f
    return decorator
