#!/usr/bin/env python3
"""
Interactive control panel for the myCobot stack.

    python3 scripts/control_panel.py

One menu for the things you actually do while the stack is running, instead of
remembering four `ros2 service call` incantations and retyping them into a
second terminal.

WHY THIS IS A NODE AND NOT A PILE OF `ros2 service call`s

Every `ros2 service call` spins up a whole node, discovers the graph, calls,
and tears down -- one to two seconds each on a loaded VM. This keeps a single
node alive with its clients already connected, so a menu choice acts
immediately. It also means the status and rate views can just watch topics
rather than shelling out.

Deliberately does NOT subscribe to /camera/image_raw. camera_node publishes
image and camera_info from the same timer callback, so their rates are
identical -- watching camera_info measures the frame rate for ~100 bytes a
message instead of ~900KB. Adding a second heavyweight image subscriber to
diagnose a host that is starved for CPU would be self-defeating.

Only needs /opt/ros/humble sourced; everything here is std_srvs and
sensor_msgs, no workspace messages.
"""

import os
import socket
import subprocess
import sys
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CameraInfo, JointState
from std_srvs.srv import SetBool, Trigger


BOLD = '\033[1m'
DIM = '\033[2m'
RED = '\033[1;31m'
GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
CYAN = '\033[1;36m'
OFF = '\033[0m'

JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']

EXPECTED_NODES = [
    'mycobot_hardware_node',
    'camera_node',
    'hand_tracker_node',
    'visual_servo_node',
    'robot_state_publisher',
]


class ControlPanel(Node):

    def __init__(self):
        super().__init__('control_panel')

        self._search = self.create_client(Trigger, '/servo/search')
        self._home = self.create_client(SetBool, '/arm/home')
        self._servo_enable = self.create_client(SetBool, '/servo/enable')
        self._jog_enable = self.create_client(SetBool, '/arm/jog_enable')

        self._joints = None
        self._joints_at = 0.0
        self.create_subscription(
            JointState, '/joint_states', self._joint_cb, 10)

        # Counters for the rate view. Both topics are tiny.
        self._cam_count = 0
        self._point_count = 0
        # camera_node publishes CameraInfo RELIABLE, /hand/point_px too.
        self.create_subscription(
            CameraInfo, '/camera/camera_info', self._cam_cb, 10)
        self.create_subscription(
            PointStamped, '/hand/point_px', self._point_cb, 10)

    # ---- callbacks ----

    def _joint_cb(self, msg: JointState):
        self._joints = msg
        self._joints_at = time.monotonic()

    def _cam_cb(self, _msg):
        self._cam_count += 1

    def _point_cb(self, _msg):
        self._point_count += 1

    # ---- helpers ----

    def _call(self, client, request, label, timeout=6.0):
        """Call a service and report, without ever hanging the menu."""
        if not client.wait_for_service(timeout_sec=1.5):
            print(f'{RED}unavailable{OFF}  {label}: service not found. '
                  'Is the stack running?')
            return None
        future = client.call_async(request)
        deadline = time.monotonic() + timeout
        while rclpy.ok() and not future.done():
            if time.monotonic() > deadline:
                print(f'{RED}timeout{OFF}  {label}: no response in '
                      f'{timeout:.0f}s')
                return None
            time.sleep(0.05)
        result = future.result()
        if result is None:
            print(f'{RED}failed{OFF}  {label}')
            return None
        ok = getattr(result, 'success', True)
        msg = getattr(result, 'message', '')
        colour = GREEN if ok else YELLOW
        print(f'{colour}{"ok" if ok else "refused"}{OFF}  {label}'
              + (f' -- {msg}' if msg else ''))
        return result

    # ---- actions ----

    def do_search(self):
        self._call(self._search, Trigger.Request(), 'search for a hand')

    def do_home(self):
        req = SetBool.Request()
        req.data = True
        print(f'{DIM}homing takes a few seconds...{OFF}')
        self._call(self._home, req, 'home the arm', timeout=25.0)

    def do_servo(self, on: bool):
        req = SetBool.Request()
        req.data = on
        self._call(self._servo_enable, req,
                   f'servoing {"on" if on else "OFF"}')

    def do_jog(self, on: bool):
        req = SetBool.Request()
        req.data = on
        self._call(self._jog_enable, req,
                   f'jogging {"on" if on else "OFF"}')

    def do_status(self):
        print(f'\n{BOLD}Nodes{OFF}')
        try:
            live = set(self.get_node_names())
        except Exception as e:
            print(f'  {RED}could not list nodes: {e}{OFF}')
            live = set()
        for name in EXPECTED_NODES:
            mark = f'{GREEN}up{OFF}' if name in live else f'{RED}DOWN{OFF}'
            print(f'  {mark:>16}  {name}')

        print(f'\n{BOLD}Joint angles{OFF}')
        if self._joints is None:
            print(f'  {RED}no /joint_states received{OFF} -- driver down, or '
                  'not connected to the arm')
        else:
            age = time.monotonic() - self._joints_at
            degs = [f'{p * 57.2958:7.1f}' for p in self._joints.position]
            print('  ' + '  '.join(
                f'{DIM}{n}{OFF}{d}' for n, d in zip(
                    [f'{j[-1]}:' for j in JOINT_NAMES], degs)))
            stale = f'{RED}(stale, {age:.1f}s){OFF}' if age > 2.0 else \
                    f'{DIM}({age:.1f}s ago){OFF}'
            print(f'  {stale}')
        print()

    def do_rates(self, seconds=5.0):
        print(f'\n{DIM}sampling for {seconds:.0f}s...{OFF}')
        self._cam_count = 0
        self._point_count = 0
        time.sleep(seconds)
        cam = self._cam_count / seconds
        pts = self._point_count / seconds

        print(f'\n{BOLD}Pipeline rates{OFF}')
        print(f'  camera frames   {cam:5.1f} Hz   {DIM}/camera/camera_info'
              f' (same rate as image_raw){OFF}')
        print(f'  detections      {pts:5.1f} Hz   {DIM}/hand/point_px{OFF}')

        if cam < 1.0:
            print(f'  {RED}No frames. camera_node is down or the Pi stream '
                  f'is dead.{OFF}')
        elif pts < 0.5:
            print(f'  {YELLOW}Frames arriving but no detections -- MediaPipe '
                  f'sees no hand. Hold one in view.{OFF}')
        elif pts < cam * 0.6:
            print(f'  {YELLOW}Detection rate is well under the frame rate, so '
                  f'inference is the bottleneck.{OFF}')
            print(f'  {DIM}The servo loop acts once per detection, so this is '
                  f'your real control rate.{OFF}')
        else:
            print(f'  {GREEN}Detection is keeping up with the camera.{OFF}')
        print()

    def do_health(self):
        ip = os.environ.get('MYCOBOT_IP', '192.168.0.15')
        print(f'\n{BOLD}Pi health ({ip}){OFF}')

        ok = subprocess.run(
            ['ping', '-c1', '-W2', ip],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        print(f'  {"ping".ljust(22)}{GREEN + "ok" + OFF if ok else RED + "FAILED" + OFF}')
        if not ok:
            print(f'  {DIM}Set MYCOBOT_IP if the address changed.{OFF}\n')
            return

        # Camera: stateless HTTP, safe to poll.
        try:
            import urllib.request
            with urllib.request.urlopen(
                    f'http://{ip}:8080/?action=snapshot', timeout=3) as r:
                cam_ok = r.status == 200
        except Exception:
            cam_ok = False
        print(f'  {"camera :8080".ljust(22)}'
              f'{GREEN + "ok" + OFF if cam_ok else RED + "FAILED" + OFF}')

        # Arm: server.py is listen(1), SINGLE CLIENT. Connect and close at
        # once -- anything that lingers here locks the driver out of the arm.
        try:
            socket.create_connection((ip, 9000), timeout=2).close()
            arm_ok = True
        except Exception:
            arm_ok = False
        note = '' if arm_ok else f'  {DIM}(may just mean the driver already ' \
                                 f'holds the single client slot){OFF}'
        print(f'  {"arm :9000".ljust(22)}'
              f'{GREEN + "ok" + OFF if arm_ok else YELLOW + "no" + OFF}{note}')
        print()


MENU = f"""
{BOLD}myCobot control panel{OFF}
  {CYAN}1{OFF}) Search for a hand      {DIM}start hunting{OFF}
  {CYAN}2{OFF}) Home the arm           {DIM}return to home pose{OFF}
  {CYAN}3{OFF}) Stop servoing          {DIM}halt the arm where it is{OFF}

  {CYAN}4{OFF}) Resume servoing        {DIM}re-arm after a stop{OFF}
  {CYAN}5{OFF}) Jogging ON  {DIM}/ {OFF}{CYAN}6{OFF}) OFF   {DIM}driver deadman{OFF}

  {CYAN}7{OFF}) Status                 {DIM}nodes + joint angles{OFF}
  {CYAN}8{OFF}) Pipeline rates         {DIM}camera vs detection Hz{OFF}
  {CYAN}9{OFF}) Health check           {DIM}Pi ping + ports 9000/8080{OFF}

  {CYAN}q{OFF}) Quit
"""


def main():
    rclpy.init()
    node = ControlPanel()

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # Let discovery settle so the first menu draw is not racing the graph.
    time.sleep(1.0)

    actions = {
        '1': node.do_search,
        '2': node.do_home,
        '3': lambda: node.do_servo(False),
        '4': lambda: node.do_servo(True),
        '5': lambda: node.do_jog(True),
        '6': lambda: node.do_jog(False),
        '7': node.do_status,
        '8': node.do_rates,
        '9': node.do_health,
    }

    try:
        while rclpy.ok():
            print(MENU)
            try:
                choice = input('choice> ').strip().lower()
            except EOFError:
                break
            if choice in ('q', 'quit', 'exit'):
                break
            action = actions.get(choice)
            if action is None:
                if choice:
                    print(f'{YELLOW}not a choice: {choice}{OFF}')
                continue
            try:
                action()
            except Exception as e:
                print(f'{RED}error: {e}{OFF}')
    except KeyboardInterrupt:
        pass
    finally:
        print('\nbye')
        executor.shutdown()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    sys.exit(main() or 0)
