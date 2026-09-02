"""Real-robot client for the openpi policy server (see start_robot_server.sh).

Runs on the ROBOT's computer, not the machine hosting the VLA. Each control
step it: reads the current language instruction, captures a fresh
observation, sends it to the remote pi0.5 policy over websocket, and
executes the returned action chunk on the robot, replanning every
`--replan-steps` steps (matching the eval_template.py sim harness this
pipeline already uses).

This is a TEMPLATE: the three functions marked TODO are stubbed out because
they depend on your robot's actual camera/proprioception/actuator APIs, none
of which exist on this machine. Fill those in before running for real.

IMPORTANT: the server this talks to is currently configured (per your setup)
to serve the `pi05_libero` checkpoint, which was trained purely on LIBERO
*simulation* data -- specific camera framing, a 224x224 padded image size, an
8-dim state (eef_pos + axis-angle + gripper_qpos) and a 7-dim delta-eef
action space at LIBERO's control rate. Actions coming back from that
checkpoint will only be meaningful on real hardware if your observation
matches that convention and something on the execute_action side maps the
output correctly (e.g. via IK) to your robot -- this script does not verify
that for you. Start with --dry-run to inspect the actions being predicted
before wiring up execute_action for real, and keep an e-stop within reach.

Install the client deps on the robot computer first:
    cd openpi/packages/openpi-client && pip install -e .

Usage:
    python robot_client.py --host <server-ip> --port 8000
"""

import argparse
import collections
import threading
import time

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy


def get_camera_frames() -> tuple[np.ndarray, np.ndarray]:
    """TODO: capture (main_image, wrist_image) as HxWx3 uint8 RGB arrays.

    Wire this up to your robot's actual camera driver(s). The images do not
    need to be pre-resized -- resize_with_pad below matches training-time
    preprocessing.
    """
    raise NotImplementedError("Wire up get_camera_frames() to your robot's cameras.")


def get_robot_state() -> np.ndarray:
    """TODO: return the 8-dim proprioceptive state pi05_libero expects:

        [eef_x, eef_y, eef_z, ax, ay, az, gripper_q1, gripper_q2]

    where (ax, ay, az) is the end-effector orientation as an axis-angle
    vector (see LIBERO-X's eval_template.py:_quat2axisangle for the
    conversion if your robot reports a quaternion instead).
    """
    raise NotImplementedError("Wire up get_robot_state() to your robot's proprioception.")


def execute_action(action: np.ndarray, *, dry_run: bool) -> None:
    """TODO: send one action to the robot's controller.

    `action` is a 7-dim vector: delta end-effector pose (dx, dy, dz, drx,
    dry, drz) followed by a gripper command, in the convention pi05_libero
    was trained on. You are responsible for mapping this to your robot's
    actual control API (e.g. via IK), including any unit/frame conversion
    and safety clamping.
    """
    if dry_run:
        print(f"[dry-run] would execute action: {np.array2string(action, precision=4)}")
        return
    raise NotImplementedError("Wire up execute_action() to your robot's controller.")


class InstructionSource:
    """Holds the current language instruction, updated from a background thread.

    Default implementation reads lines from stdin, so you can type a new
    command at any time between control steps. Replace `_listen` with
    whatever channel actually delivers commands on your robot (a ROS topic
    subscriber, an MQTT callback, a socket, etc.) -- the rest of the script
    only depends on `.current` being kept up to date.
    """

    def __init__(self, initial: str):
        self._lock = threading.Lock()
        self._current = initial
        thread = threading.Thread(target=self._listen, daemon=True)
        thread.start()

    def _listen(self) -> None:
        import sys

        for line in sys.stdin:
            line = line.strip()
            if line:
                with self._lock:
                    self._current = line
                print(f"[instruction updated] {line}")

    @property
    def current(self) -> str:
        with self._lock:
            return self._current


def main(args: argparse.Namespace) -> None:
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    print(f"Connected. Server metadata: {client.get_server_metadata()}")

    instructions = InstructionSource(args.initial_instruction)
    if args.initial_instruction == "":
        print("No --initial-instruction given -- type a command and press enter to start.")

    action_plan: collections.deque = collections.deque()
    step_period = 1.0 / args.control_hz

    print("Starting control loop. Ctrl+C to stop.")
    try:
        while True:
            step_start = time.monotonic()

            if not action_plan:
                main_img, wrist_img = get_camera_frames()
                main_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(main_img, args.resize_size, args.resize_size)
                )
                wrist_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                )
                state = get_robot_state()
                prompt = instructions.current

                observation = {
                    "observation/image": main_img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": state,
                    "prompt": prompt,
                }
                response = client.infer(observation)
                action_chunk = response["actions"]
                if len(action_chunk) == 0:
                    raise RuntimeError("Policy returned zero actions.")
                chunk_len = len(action_chunk) if args.replan_steps is None else min(len(action_chunk), args.replan_steps)
                action_plan.extend(action_chunk[:chunk_len])
                print(
                    f"Replanned ({prompt!r}): {chunk_len} steps queued, "
                    f"infer_ms={response.get('server_timing', {}).get('infer_ms', float('nan')):.1f}"
                )

            action = action_plan.popleft()
            execute_action(np.asarray(action), dry_run=args.dry_run)

            elapsed = time.monotonic() - step_start
            time.sleep(max(0.0, step_period - elapsed))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", required=True, help="IP of the machine hosting the policy server.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--resize-size", type=int, default=224, help="Must match the policy's training resolution.")
    parser.add_argument(
        "--replan-steps",
        type=int,
        default=None,
        help="Re-query the policy every N executed steps instead of running the full predicted chunk open-loop. "
        "Default (unset) plays the full chunk returned by the server (10 steps for pi05_libero) before replanning.",
    )
    parser.add_argument("--control-hz", type=float, default=10.0, help="Target action execution rate.")
    parser.add_argument("--initial-instruction", type=str, default="", help="Language command to start with.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print predicted actions instead of executing them on the robot. Use this first.",
    )
    main(parser.parse_args())
