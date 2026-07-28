#!/usr/bin/env python3
"""
The one script. Brings the stack up and gives you a menu to drive it.

    ./run.py                    launch everything, then show the menu
    ./run.py gain:=1.5          extra args pass through to ros2 launch

If the stack is already running it attaches to it instead of starting a
second one.

HOW IT IS PUT TOGETHER

The launch output would drown a menu sharing the same terminal, so the stack
is started detached with its console redirected to a log file, and the menu
runs in the foreground. The log stays one keypress away -- 'l' prints the
last 40 lines and comes straight back, 'f' follows it live -- because that
output is how you actually diagnose this system.

Service calls go through a single long-lived node rather than shelling out to
`ros2 service call`, which spins up a node, discovers the graph, calls, and
tears down on every invocation: one to two seconds each on a loaded VM. Here
the clients stay connected and a menu choice acts immediately.

Quitting shuts the stack down cleanly (SIGINT to the process group, which is
what ros2 launch expects), so you do not leave nodes holding the arm's single
client slot.

Requires ROS 2 sourced, which the README setup puts in your .bashrc.
"""

import os
import signal
import socket
import subprocess
import sys
import threading
import time

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import MultiThreadedExecutor
    from geometry_msgs.msg import PointStamped
    from sensor_msgs.msg import CameraInfo, JointState
    from std_srvs.srv import SetBool, Trigger
except ImportError as e:
    sys.stderr.write(
        f'\033[1;31m[error]\033[0m Could not import ROS 2 Python ({e}).\n'
        '  Source it first:  source /opt/ros/humble/setup.bash\n'
        '  and the workspace: source install/setup.bash\n')
    sys.exit(1)


BOLD, DIM = '\033[1m', '\033[2m'
RED, GREEN, YELLOW, CYAN = '\033[1;31m', '\033[1;32m', '\033[1;33m', '\033[1;36m'
OFF = '\033[0m'

REPO = os.path.dirname(os.path.abspath(__file__))
LOG = '/tmp/mycobot_stack.log'
IP = os.environ.get('MYCOBOT_IP', '192.168.0.15')

JOINT_LABELS = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6']
EXPECTED_NODES = [
    'mycobot_hardware_node', 'camera_node',
    'hand_tracker_node', 'visual_servo_node', 'robot_state_publisher',
]


def say(m):  print(f'{CYAN}==>{OFF} {m}')
def warn(m): print(f'{YELLOW}[warn]{OFF} {m}')
def die(m):  print(f'{RED}[error]{OFF} {m}', file=sys.stderr); sys.exit(1)


# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------

def port_open(host, port, timeout=2.0):
    """Connect and close AT ONCE.

    server.py on :9000 is listen(1) -- single client. Anything that holds this
    socket open locks the driver out of the arm for as long as it lives.
    """
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except Exception:
        return False


def ping_ok(host, count=3, wait=3):
    """Best-effort reachability hint.

    Deliberately more than one packet: on a cold ARP cache -- which is exactly
    the state after power-cycling the Pi -- the first ICMP is routinely
    dropped while resolution happens, so `ping -c1` reports a healthy host as
    unreachable. This is only a hint either way; the port checks decide.
    """
    return subprocess.run(
        ['ping', '-c', str(count), '-W', str(wait), host],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def camera_ok(host, timeout=3.0):
    try:
        import urllib.request
        with urllib.request.urlopen(
                f'http://{host}:8080/?action=snapshot', timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


PI_FILES = ['server.py', 'camera_stream.py',
            'mycobot_server.service', 'mjpg_streamer.service']


def pi_files_stale():
    """Which pi/ files on the robot differ from this repo.

    The Pi runs its own copy of server.py and camera_stream.py, pushed there
    by scripts/redeploy_pi.sh. Editing them here does nothing until they are
    deployed -- and because the arm-side fixes (the client idle timeout, for
    one) live entirely in that copy, it is entirely possible to "fix" a
    disconnect, rebuild the workspace, and change nothing at all about the
    robot's behaviour.

    Returns a list of stale filenames, [] if in sync, or None if it could not
    be determined (no ssh key, different layout).
    """
    import hashlib
    user = os.environ.get('MYCOBOT_PI_USER', 'er')
    pidir = os.environ.get('MYCOBOT_PI_DIR', '~/JON/mycobot_project/pi')

    local = {}
    for f in PI_FILES:
        p = os.path.join(REPO, 'pi', f)
        if os.path.isfile(p):
            with open(p, 'rb') as fh:
                local[f] = hashlib.md5(fh.read()).hexdigest()
    if not local:
        return None

    remote_paths = ' '.join(f'{pidir}/{f}' for f in local)
    r = subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
         f'{user}@{IP}', f'md5sum {remote_paths} 2>/dev/null'],
        capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        return None

    remote = {}
    for line in r.stdout.strip().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            remote[os.path.basename(parts[1])] = parts[0]
    return sorted(f for f, h in local.items() if remote.get(f) != h)


def check_pi_in_sync():
    stale = pi_files_stale()
    if stale is None:
        print(f'{DIM}    (could not verify the Pi is running current code){OFF}')
        return
    if not stale:
        say('Pi files in sync.')
        return

    print(f'\n{RED}{"=" * 66}{OFF}')
    print(f'{RED}  The robot is running OUTDATED code.{OFF}')
    print(f'  These differ from this repo: {BOLD}{", ".join(stale)}{OFF}')
    print(f'  Changes to pi/ do nothing until they are pushed to the robot.')
    print(f'{RED}{"=" * 66}{OFF}\n')
    try:
        ans = input('Redeploy to the Pi now? [Y/n] ').strip().lower()
    except EOFError:
        ans = 'n'
    if ans in ('', 'y', 'yes'):
        rc = subprocess.run(
            [os.path.join(REPO, 'scripts', 'redeploy_pi.sh'), IP],
            cwd=REPO).returncode
        if rc != 0:
            warn('Redeploy failed; continuing with the old Pi code.')
        else:
            say('Redeployed.')
            time.sleep(2)
    else:
        warn('Continuing with outdated Pi code.')


def preflight():
    """Prove the Pi is serving both ports before launching.

    Without this, an unreachable arm surfaces as 'Waiting for /arm/jog_enable'
    and a dead camera as a servo node that never sees a frame -- two confusing
    ROS-level symptoms for one plain infrastructure problem.
    """
    say(f'Checking Pi at {IP} ...')
    if ping_ok(IP):
        say('Pi reachable.')
    else:
        # Never fatal. ICMP is not what this stack needs -- the two ports are
        # -- and plenty of things make ping lie: a cold ARP cache, a firewall
        # dropping ICMP, a switch still learning. Failing here would refuse to
        # launch against a Pi that is serving both ports perfectly well.
        warn(f'No ping response from {IP}. Continuing anyway -- the port '
             'checks below are what actually matter.')

    check_pi_in_sync()

    say('Waiting for camera on :8080 ...')
    for i in range(30):
        if camera_ok(IP):
            say('Camera OK')
            break
        if i == 29:
            die(f'Camera never came up on {IP}:8080.\n'
                f'  If ping also failed, the Pi is off or on another address '
                f'(set MYCOBOT_IP).\n'
                f"  If ping worked, the service is down: "
                f"ssh {os.environ.get('MYCOBOT_PI_USER', 'er')}@{IP} "
                f"'systemctl status mjpg_streamer'")
        time.sleep(1)

    say('Waiting for arm TCP on :9000 ...')
    for i in range(30):
        if port_open(IP, 9000):
            say('Arm port OK')
            break
        if i == 29:
            die(f'Arm never came up on {IP}:9000.\n'
                f"  ssh {os.environ.get('MYCOBOT_PI_USER', 'er')}@{IP} "
                f"'systemctl status mycobot_server'\n"
                '  Note server.py accepts ONE client -- a leftover '
                'measure_arm.py or a second driver holds the slot and this '
                'check will keep failing until it exits.')
        time.sleep(1)
    # Let server.py finish closing the probe before the driver claims the slot.
    time.sleep(1)


def ensure_built():
    if not os.path.isfile(os.path.join(REPO, 'install', 'setup.bash')):
        say('No build found, running colcon build (first run only)...')
        if subprocess.run(['colcon', 'build', '--symlink-install'],
                          cwd=REPO).returncode != 0:
            die('colcon build failed.')
        warn('Built. Re-run after: source install/setup.bash')
        sys.exit(0)


# --------------------------------------------------------------------------
# Stack process
# --------------------------------------------------------------------------

class Stack:
    def __init__(self, extra_args):
        self.extra = extra_args
        self.proc = None

    def start(self):
        env = dict(os.environ, MYCOBOT_IP=IP)
        cmd = ['ros2', 'launch', 'mycobot_bringup',
               'servo_demo.launch.py'] + self.extra
        say(f'Launching (MYCOBOT_IP={IP}), output -> {LOG}')
        self.log = open(LOG, 'w')
        # start_new_session so the whole launch tree can be signalled as one
        # group, and so Ctrl-C in this menu does not tear the stack down by
        # accident.
        self.proc = subprocess.Popen(
            cmd, cwd=REPO, env=env, stdout=self.log,
            stderr=subprocess.STDOUT, start_new_session=True)

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        if not self.alive():
            return
        say('Shutting the stack down...')
        try:
            # SIGINT is what ros2 launch expects for a clean shutdown; it
            # propagates to the nodes so they release the arm socket properly.
            os.killpg(os.getpgid(self.proc.pid), signal.SIGINT)
        except ProcessLookupError:
            return
        for _ in range(100):
            if not self.alive():
                break
            time.sleep(0.1)
        if self.alive():
            warn('Did not exit on SIGINT, terminating.')
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            self.log.close()
        except Exception:
            pass


# --------------------------------------------------------------------------
# Control node
# --------------------------------------------------------------------------

class Panel(Node):
    def __init__(self):
        super().__init__('run_control_panel')
        self._search = self.create_client(Trigger, '/servo/search')
        self._home = self.create_client(SetBool, '/arm/home')
        self._servo = self.create_client(SetBool, '/servo/enable')
        self._jog = self.create_client(SetBool, '/arm/jog_enable')

        self._joints = None
        self._joints_at = 0.0
        self._cam_n = 0
        self._pt_n = 0

        self.create_subscription(JointState, '/joint_states', self._j_cb, 10)
        # camera_node publishes image and camera_info from the same timer
        # callback, so their rates are identical -- watching camera_info
        # measures the frame rate for ~100 bytes instead of ~900KB. Adding a
        # heavyweight image subscriber to diagnose a CPU-starved host would
        # be self-defeating.
        self.create_subscription(CameraInfo, '/camera/camera_info',
                                 self._c_cb, 10)
        self.create_subscription(PointStamped, '/hand/point_px',
                                 self._p_cb, 10)

    def _j_cb(self, m): self._joints, self._joints_at = m, time.monotonic()
    def _c_cb(self, _): self._cam_n += 1
    def _p_cb(self, _): self._pt_n += 1

    def _call(self, client, req, label, timeout=6.0):
        if not client.wait_for_service(timeout_sec=1.5):
            print(f'{RED}unavailable{OFF}  {label}: service not found.')
            return None
        fut = client.call_async(req)
        end = time.monotonic() + timeout
        while rclpy.ok() and not fut.done():
            if time.monotonic() > end:
                print(f'{RED}timeout{OFF}  {label}')
                return None
            time.sleep(0.05)
        res = fut.result()
        if res is None:
            print(f'{RED}failed{OFF}  {label}')
            return None
        ok = getattr(res, 'success', True)
        msg = getattr(res, 'message', '')
        print(f'{GREEN if ok else YELLOW}{"ok" if ok else "refused"}{OFF}  '
              f'{label}' + (f' -- {msg}' if msg else ''))
        return res

    def wait_for_nodes(self, timeout=45.0):
        say('Waiting for nodes...')
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            live = set(self.get_node_names())
            if all(n in live for n in EXPECTED_NODES):
                say('All nodes up.')
                return True
            time.sleep(1.0)
        warn('Not all nodes appeared; check the log with "l".')
        return False

    # ---- actions ----

    def search(self):
        self._call(self._search, Trigger.Request(), 'search for a hand')

    def home(self):
        r = SetBool.Request(); r.data = True
        print(f'{DIM}homing takes a few seconds...{OFF}')
        self._call(self._home, r, 'home the arm', timeout=25.0)

    def servo(self, on):
        r = SetBool.Request(); r.data = on
        self._call(self._servo, r, f'servoing {"on" if on else "OFF"}')

    def jog(self, on):
        r = SetBool.Request(); r.data = on
        self._call(self._jog, r, f'jogging {"on" if on else "OFF"}')

    def status(self):
        print(f'\n{BOLD}Nodes{OFF}')
        try:
            live = set(self.get_node_names())
        except Exception as e:
            print(f'  {RED}could not list nodes: {e}{OFF}'); live = set()
        for n in EXPECTED_NODES:
            print(f'  {(GREEN + "up" + OFF) if n in live else (RED + "DOWN" + OFF):>16}  {n}')

        print(f'\n{BOLD}Joint angles{OFF}')
        if self._joints is None:
            print(f'  {RED}no /joint_states{OFF} -- driver down, or not '
                  'connected to the arm')
        else:
            age = time.monotonic() - self._joints_at
            cells = '  '.join(
                f'{DIM}{lab}{OFF}{p * 57.2958:7.1f}'
                for lab, p in zip(JOINT_LABELS, self._joints.position))
            print(f'  {cells}')
            print(f'  {(RED + f"(stale {age:.1f}s)" + OFF) if age > 2 else (DIM + f"({age:.1f}s ago)" + OFF)}')
        print()

    def rates(self, seconds=5.0):
        print(f'\n{DIM}sampling {seconds:.0f}s...{OFF}')
        self._cam_n = self._pt_n = 0
        time.sleep(seconds)
        cam, pts = self._cam_n / seconds, self._pt_n / seconds
        print(f'\n{BOLD}Pipeline rates{OFF}')
        print(f'  camera frames {cam:5.1f} Hz')
        print(f'  detections    {pts:5.1f} Hz')
        if cam < 1:
            print(f'  {RED}No frames -- camera_node down or Pi stream dead.{OFF}')
        elif pts < 0.5:
            print(f'  {YELLOW}Frames but no detections -- hold a hand in view.{OFF}')
        elif pts < cam * 0.6:
            print(f'  {YELLOW}Detection well under frame rate: inference is '
                  f'the bottleneck.{OFF}')
            print(f'  {DIM}The servo acts once per detection, so this is your '
                  f'real control rate.{OFF}')
        else:
            print(f'  {GREEN}Detection keeping up with the camera.{OFF}')
        print()

    def health(self):
        print(f'\n{BOLD}Pi health ({IP}){OFF}')
        ok = ping_ok(IP, count=2)
        # Report but never stop here -- ping failing while both ports answer
        # is a normal outcome (ICMP filtered, cold ARP), and the ports are the
        # part that matters.
        print(f'  {"ping".ljust(20)}'
              f'{GREEN + "ok" + OFF if ok else YELLOW + "no reply" + OFF}')
        c = camera_ok(IP)
        print(f'  {"camera :8080".ljust(20)}{GREEN + "ok" + OFF if c else RED + "FAILED" + OFF}')
        a = port_open(IP, 9000)
        note = '' if a else f'  {DIM}(usually just means the driver holds the single client slot){OFF}'
        print(f'  {"arm :9000".ljust(20)}{GREEN + "ok" + OFF if a else YELLOW + "no" + OFF}{note}')
        print()


# --------------------------------------------------------------------------

def show_log(lines=40):
    """Print the tail of the log and return to the menu immediately.

    Deliberately not `tail -f`: following blocks the menu, and getting back
    out meant Ctrl-C, which the terminal delivers to the whole foreground
    process group -- killing the stack along with the tail. A snapshot you
    can take repeatedly is more useful than a stream you cannot leave.
    """
    print(f'{DIM}--- last {lines} lines of {LOG} ---{OFF}')
    try:
        subprocess.run(['tail', '-n', str(lines), LOG])
    except Exception as e:
        print(f'{RED}could not read {LOG}: {e}{OFF}')
    print(f'{DIM}--- end (press l again to refresh, f to follow live) ---{OFF}\n')


def follow_log():
    """Stream the log until Ctrl-C, then return to the menu.

    tail runs in its own session so the terminal's Ctrl-C reaches only this
    process. Without that the signal goes to the entire foreground group and
    takes the stack down with it.
    """
    print(f'{DIM}--- following {LOG} (Ctrl-C returns to the menu) ---{OFF}')
    p = subprocess.Popen(['tail', '-n', '20', '-f', LOG],
                         start_new_session=True)
    try:
        p.wait()
    except KeyboardInterrupt:
        print(f'\n{DIM}--- stopped following ---{OFF}')
    finally:
        p.terminate()
        try:
            p.wait(timeout=2)
        except Exception:
            p.kill()
    print()


def menu(attached):
    quit_note = 'Quit (leaves the stack running)' if attached else \
                'Quit (shuts the stack down)'
    return f"""
{BOLD}myCobot control panel{OFF}
  {CYAN}1{OFF}) Search for a hand      {DIM}start hunting{OFF}
  {CYAN}2{OFF}) Home the arm           {DIM}return to home pose{OFF}
  {CYAN}3{OFF}) Stop servoing          {DIM}halt the arm where it is{OFF}

  {CYAN}4{OFF}) Resume servoing        {DIM}re-arm after a stop{OFF}
  {CYAN}5{OFF}) Jogging ON  {DIM}/{OFF} {CYAN}6{OFF}) OFF   {DIM}driver deadman{OFF}

  {CYAN}7{OFF}) Status                 {DIM}nodes + joint angles{OFF}
  {CYAN}8{OFF}) Pipeline rates         {DIM}camera vs detection Hz{OFF}
  {CYAN}9{OFF}) Health check           {DIM}Pi ping + ports 9000/8080{OFF}
  {CYAN}l{OFF}) Show log               {DIM}last 40 lines, returns straight back{OFF}
  {CYAN}f{OFF}) Follow log             {DIM}live stream, Ctrl-C to come back{OFF}

  {CYAN}q{OFF}) {quit_note}
"""


def main():
    args = [a for a in sys.argv[1:] if a not in ('-h', '--help')]
    if len(args) != len(sys.argv[1:]):
        print(__doc__)
        return 0

    ensure_built()

    rclpy.init()
    panel = Panel()
    ex = MultiThreadedExecutor(num_threads=2)
    ex.add_node(panel)
    threading.Thread(target=ex.spin, daemon=True).start()
    time.sleep(1.0)  # let discovery settle

    # Attach rather than starting a second stack on top of a running one --
    # two drivers would fight over the arm's single client slot.
    attached = any(n in set(panel.get_node_names())
                   for n in ('mycobot_hardware_node', 'visual_servo_node'))
    stack = None
    if attached:
        say('Stack already running -- attaching to it.')
    else:
        preflight()
        stack = Stack(args)
        stack.start()
        panel.wait_for_nodes()

    actions = {
        '1': panel.search,
        '2': panel.home,
        '3': lambda: panel.servo(False),
        '4': lambda: panel.servo(True),
        '5': lambda: panel.jog(True),
        '6': lambda: panel.jog(False),
        '7': panel.status,
        '8': panel.rates,
        '9': panel.health,
        'l': show_log,
        'f': follow_log,
    }

    try:
        while rclpy.ok():
            if stack and not stack.alive():
                warn('The stack exited. Check the log with "l".')
            print(menu(attached))
            try:
                choice = input('choice> ').strip().lower()
            except EOFError:
                break
            if choice in ('q', 'quit', 'exit'):
                break
            act = actions.get(choice)
            if act is None:
                if choice:
                    print(f'{YELLOW}not a choice: {choice}{OFF}')
                continue
            try:
                act()
            except KeyboardInterrupt:
                print()
            except Exception as e:
                print(f'{RED}error: {e}{OFF}')
    except KeyboardInterrupt:
        pass
    finally:
        ex.shutdown()
        panel.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass
        if stack:
            stack.stop()
        print('bye')
    return 0


if __name__ == '__main__':
    sys.exit(main())
