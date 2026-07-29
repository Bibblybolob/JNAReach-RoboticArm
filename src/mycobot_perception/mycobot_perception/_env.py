"""Fail early and legibly when the installed stack cannot work.

cv_bridge is a C extension compiled against NumPy 1.x and OpenCV 4. Under
NumPy 2 or OpenCV 5 the IMPORT merely prints a warning -- it is the first
actual conversion that dies, and it dies as SIGSEGV with `AttributeError:
_ARRAY_API not found` buried above it.

That failure mode is unusually expensive to debug:

  * The node logs "ready" first, because the import survived. Everything
    looks like it started correctly.
  * It then dies on the first frame, so it presents as "the camera works but
    nothing is detected" rather than as a broken install.
  * launch respawns it, so it dies repeatedly and the traceback scrolls away.
  * The error names cv_bridge, which is not what is wrong.

requirements.txt pins both bounds for exactly this reason. This module turns
the whole thing into one line at startup, before any of that can happen.
"""

REQUIREMENTS = 'pip install -r requirements.txt'


def check() -> None:
    """Raise SystemExit with a fix if the versions cannot work."""
    problems = []

    try:
        import numpy
        major = int(numpy.__version__.split('.')[0])
        if major >= 2:
            problems.append(
                f'  numpy {numpy.__version__} -- cv_bridge is compiled '
                'against NumPy 1.x and segfaults on the first frame under 2.x'
            )
    except ImportError:
        problems.append('  numpy is not installed')

    try:
        import cv2
        major = int(cv2.__version__.split('.')[0])
        if major >= 5:
            problems.append(
                f'  opencv {cv2.__version__} -- cv_bridge maps OpenCV 4 type '
                'codes; under 5 conversions return wrong or empty images'
            )
    except ImportError:
        problems.append('  opencv is not installed')

    if not problems:
        return

    raise SystemExit(
        '\n'
        'This node cannot run against the installed versions:\n\n'
        + '\n'.join(problems)
        + '\n\n'
        'These bounds are in requirements.txt and are load-bearing. Without\n'
        'this check the node starts, logs that it is ready, and then dies on\n'
        'the first camera frame with a segfault whose message names\n'
        'cv_bridge rather than the actual problem.\n\n'
        f'    {REQUIREMENTS}\n\n'
        'If both opencv-python and opencv-contrib-python are installed,\n'
        'remove the plain one first -- they overwrite each other:\n\n'
        '    pip uninstall -y opencv-python\n'
    )
