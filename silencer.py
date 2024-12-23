from sys import stdout, stderr
from os  import devnull
 
from contextlib import redirect_stdout, redirect_stderr, contextmanager


@contextmanager
def silence(out=True, err=False):
    null = open(devnull, 'w')
    out = null if out else stdout
    err = null if err else stderr

    # redireção das saídas padrão
    with redirect_stdout(out), redirect_stderr(err):
        yield out, err

    return
