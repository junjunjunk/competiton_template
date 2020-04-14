import logging

from typing import Optional


@contextmanager
def timer(name: str, logger: Optional[logging.Logger] = None):
    """
    with文内の時間を測ってくれる

    with timer("wait"):
        wait(2.0)
    """

    t0 = time.time()
    yield
    msg = f"[{name}] done in {time.time()-t0:.0f} s"
    if logger:
        logger.info(msg)
    else:
        print(msg)
