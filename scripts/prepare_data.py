import sys
from crisp_da.cli import main

if __name__ == "__main__":
    main(["make-manifests", *sys.argv[1:]])
