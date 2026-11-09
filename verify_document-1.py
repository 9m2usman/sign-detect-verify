"""
Quick-launch wrapper for Signature Verification CLI.
Ensures correct module paths regardless of which folder the terminal is in.
"""

import sys
import os

PROJECT_ROOT = "E:/sign_detect-verify"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.chdir(PROJECT_ROOT)

from verify_document import main

if __name__ == "__main__":
    main()
