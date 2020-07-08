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
        return "ERROR_GIT_NO_DIRECTORY"
    # We use git diff head because it shows all differences,
    # including any that have been staged but not committed.
    try:
        diff = subprocess.run(
            'git diff HEAD'.split(),
            cwd=dirname,
            universal_newlines=True,
            timeout=5,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    # There are lots of different ways this can go wrong.
    # Here are some - any others it is probably worth knowing
    # about
    except subprocess.TimeoutExpired:
        return "ERROR_GIT_TIMEOUT"
    except UnicodeDecodeError:
        return "ERROR_GIT_DECODING"
    except subprocess.SubprocessError:
        return "ERROR_GIT_OTHER"
    except FileNotFoundError:
        return "ERROR_GIT_NOT_RUNNABLE"
    except OSError:
        return "ERROR_GIT_OTHER_OSERROR"
    # If for some reason we are running outside the main repo
    # this will return an error too
    if diff.returncode:
        return "ERROR_GIT_FAIL"

    return diff.stdout


def git_current_revision():
    """Run git diff in the caller's directory, and return stdout+stderr
    """
    dirname = get_caller_directory(grandparent=True)
    if dirname is None:
        return "ERROR_GIT_NO_DIRECTORY"
    try:
        rev = subprocess.run(
            'git rev-parse HEAD'.split(),
            cwd=dirname,
            universal_newlines=True,
            timeout=5,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    # Same as git diff above.
    except subprocess.TimeoutExpired:
        return "ERROR_GIT_TIMEOUT"
    except UnicodeDecodeError:
        return "ERROR_GIT_DECODING"
    except subprocess.SubprocessError:
        return "ERROR_GIT_OTHER"
    except FileNotFoundError:
        return "ERROR_GIT_NOT_RUNNABLE"
    except OSError:
        return "ERROR_GIT_OTHER_OSERROR"
    # If for some reason we are running outside the main repo
    # this will return an error too
    if rev.returncode:
        return "ERROR_GIT_FAIL"
    return rev.stdout
