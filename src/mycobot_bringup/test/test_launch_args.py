#!/usr/bin/env python3
"""Launch arguments declared in more than one file must agree.

    python3 src/mycobot_bringup/test/test_launch_args.py

Runs without ROS -- it parses the launch files as text rather than executing
them, so it works anywhere the repo is checked out.

WHY THIS EXISTS

servo_demo.launch.py includes robot_bringup.launch.py and forwards `source` to
it. When `realsense` was added as a valid source, only the outer file's
`choices` list was updated. The outer file then accepted the value, passed it
inward, and the inner file rejected it:

    Argument "source" provided value "realsense" is not valid.
    Valid options are: ['mjpeg', 'device']

Nothing catches that before runtime. `run.py` validates argument NAMES but not
values. `ros2 launch --show-args` reports the OUTER file's choices, which look
correct -- it does not descend into included launch files. So the only signal
is the stack refusing to start, with an error naming a constraint you have
already fixed in the file you were looking at.
"""
import collections
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


def declarations():
    """{arg name: {launch file: choices literal}} for every declared choices."""
    found = collections.defaultdict(dict)
    pattern = glob.glob(os.path.join(ROOT, 'src', '*', 'launch', '*.py'))
    if not pattern:
        sys.exit(f'No launch files found under {ROOT}/src/*/launch/')
    for path in sorted(pattern):
        src = open(path).read()
        for m in re.finditer(
                r"DeclareLaunchArgument\(\s*'([^']+)'(.*?)\)", src, re.S):
            name, body = m.group(1), m.group(2)
            choices = re.search(r"choices=(\[[^\]]*\])", body)
            if choices:
                found[name][os.path.basename(path)] = choices.group(1)
    return found, len(pattern)


def test_choices_agree():
    found, n_files = declarations()
    shared = {k: v for k, v in found.items() if len(v) > 1}
    print(f'  {n_files} launch files, {len(found)} arguments with choices, '
          f'{len(shared)} declared in more than one file')

    bad = []
    for name, where in sorted(shared.items()):
        values = set(where.values())
        if len(values) > 1:
            bad.append((name, where))
        else:
            print(f'    ok  {name}: {next(iter(values))} '
                  f'({len(where)} files agree)')

    if bad:
        print()
        for name, where in bad:
            print(f'  MISMATCH on "{name}":')
            for f, v in sorted(where.items()):
                print(f'      {f}: {v}')
        print()
        print('  A value accepted by the outer launch file and rejected by an')
        print('  included one fails at launch, naming a constraint you have')
        print('  already fixed in the file you were reading.')
        raise AssertionError(f'{len(bad)} argument(s) disagree on choices')


def test_source_includes_realsense():
    """The specific regression: every file declaring `source` knows the three."""
    found, _ = declarations()
    where = found.get('source')
    assert where, 'no launch file declares a `source` argument any more'
    for f, choices in sorted(where.items()):
        for expected in ('mjpeg', 'device', 'realsense'):
            assert expected in choices, \
                f'{f} declares source without {expected!r}: {choices}'
        print(f'    ok  {f}: {choices}')


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    print(f'{len(tests)} tests, no ROS required\n')
    failed = 0
    for fn in tests:
        print(f'{fn.__name__}:')
        try:
            fn()
        except AssertionError as e:
            print(f'  FAILED: {e}')
            failed += 1
    print()
    print(f'All {len(tests)} passed.' if not failed
          else f'{failed} of {len(tests)} FAILED.')
    sys.exit(1 if failed else 0)
