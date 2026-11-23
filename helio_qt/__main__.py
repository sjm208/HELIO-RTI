import multiprocessing
import sys

from helio_qt.app import main

# Guarded: image loading uses a process pool, and each worker process imports
# this module under another name.  Unguarded, every worker would open a window.
if __name__ == "__main__":
    multiprocessing.freeze_support()      # a no-op from source; needed in the built app
    sys.exit(main(sys.argv))
