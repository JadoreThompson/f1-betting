import os
from multiprocessing import Queue

# To be initialised by __main__
if os.getenv("PYTEST_RUNNING"):
    MATCHING_ENGINE_QUEUE: Queue = None
else:
    MATCHING_ENGINE_QUEUE: Queue
