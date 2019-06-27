import distutils.version
import sys
import inspect
import pathlib
import subprocess

def find_module_versions():
    """
    Generate a dictionary of versions of all imported modules
    by looking for __version__ or version attributes on them.
    """
    versions = {}
    for name, module in sys.modules.items(): 
        if hasattr(module, 'version'): 
            v = module.version 
        elif hasattr(module, '__version__'): 
            v = module.__version__ 
        else:
            continue
        if isinstance(v, str) or isinstance(v, distutils.version.Version):
            versions[name] = str(v)
    return versions


def get_caller_directory(grandparent=False):
    """
    Find the directory where the code calling this
    module lives, or where the code calling
    that code lives if grandparent=True.
    """
    previous_frame = inspect.currentframe().f_back
    if grandparent:
        previous_frame = previous_frame.f_back

    filename = inspect.getframeinfo(previous_frame).filename
    p = pathlib.Path(filename)
    if not p.exists():
        # dynamically generated or interactive moed
        return None
    return str(p.parent)

def git_diff():
    """Run git diff in the caller's directory, and return stdout+stderr
    """
    dirname = get_caller_directory(grandparent=True)
    if dirname is None:
        return "ERROR_NO_GIT_DIFF_AVAILABLE"
    # We use git diff head because it shows all differences,
    # including any that have been staged but not committed.
    diff = subprocess.run('git diff HEAD'.split(),
        cwd=dirname, capture_output=True, text=True)

    # You can't interleave stdout and stderr when you do capture_output,
    # unfortunately.
    return diff.stdout + diff.stderr

def git_current_revision():
    """Run git diff in the caller's directory, and return stdout+stderr
    """
    dirname = get_caller_directory(grandparent=True)
    if dirname is None:
        return "ERROR_NO_GIT_REV_AVAILABLE"
    diff = subprocess.run('git rev-parse HEAD'.split(),
        cwd=dirname, capture_output=True, text=True)
    return diff.stdout + diff.stderr

