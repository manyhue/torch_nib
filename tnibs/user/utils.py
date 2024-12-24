# utils, useful for user and implementation

import inspect
import os


def get_project_name():
    import subprocess

    repo_name = (
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
        .decode("utf-8")
        .strip()
    )
    repo_name = os.path.basename(repo_name)

    caller_frame = inspect.stack()[1]
    caller_file = caller_frame.filename
    filename = os.path.splitext(os.path.basename(caller_file))[0]
    return f"{repo_name}_{filename}"
