"""
gazebo_bridge.py — Gazebo + PX4 SITL interface for SeaPatch Phase 3.

Provides the interface between the SeaPatch ground station and a Gazebo
simulation running 40 drones via PX4 SITL + ROS2.

Phase 2 does NOT require this file — use SeaDronesSeebridge (seadronessee_bridge.py)
for frame injection directly from the dataset. This bridge is for Phase 3 when live
rendered camera feeds replace dataset frame injection.

See DESIGN.md §5.1 for the full Gazebo + PX4 SITL setup instructions.

Prerequisites (Phase 3):
    ROS2 Humble or Jazzy
    PX4 Autopilot (main branch)
    MAVROS or px4_ros_com for ROS2↔PX4 bridge
    Gazebo Garden or Harmonic
    pip install rclpy sensor_msgs geometry_msgs px4_msgs

Usage:
    # Start PX4 SITL + Gazebo first (outside Python):
    #   cd PX4-Autopilot && make px4_sitl gazebo-classic_iris_depth_camera
    # Then:
    bridge = GazeboBridge(n_drones=40)
    await bridge.connect()
    frame_path = await bridge.get_camera_frame("S22")
    pose = await bridge.get_drone_pose("S22")
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any


class GazeboBridgeError(Exception):
    pass


class GazeboBridge:
    """Interface to Gazebo + PX4 SITL for live drone camera feeds.

    Phase 3 only. Phase 2 uses SeaDronesSeebridge instead.

    Drone IDs S01-S38 map to PX4 instances 1-38.
    Drone IDs C1-C2 map to PX4 instances 39-40.
    """

    def __init__(
        self,
        n_scouts: int = 38,
        n_commanders: int = 2,
        ros_namespace: str = "/seapatch",
        frame_save_dir: Path | None = None,
    ) -> None:
        self._n_scouts = n_scouts
        self._n_commanders = n_commanders
        self._ros_namespace = ros_namespace
        self._frame_save_dir = frame_save_dir or Path(tempfile.mkdtemp(prefix="seapatch_frames_"))
        self._connected = False
        self._node: Any = None  # rclpy.Node — imported lazily

    async def connect(self) -> None:
        """Initialise ROS2 node and subscribe to drone camera topics.

        Raises GazeboBridgeError if ROS2 is not available.
        This is a Phase 3 stub — raises NotImplementedError until implemented.
        """
        raise NotImplementedError(
            "GazeboBridge.connect() is a Phase 3 stub. "
            "Use SeaDronesSeebridge for Phase 1/2. "
            "See DESIGN.md §5.1 for Phase 3 setup."
        )

    async def get_camera_frame(
        self,
        drone_id: str,
        channel: str = "rgb",
        timeout_s: float = 5.0,
    ) -> Path:
        """Subscribe to a drone's camera topic and save the latest frame to disk.

        Parameters
        ----------
        drone_id:
            Scout or commander drone ID (e.g. "S22", "C1").
        channel:
            "rgb" (all drones) or "thermal" (commanders only).
        timeout_s:
            How long to wait for a frame before raising.

        Returns
        -------
        Path to the saved JPEG frame.

        Raises
        ------
        GazeboBridgeError if not connected or frame times out.
        NotImplementedError — Phase 3 stub.
        """
        self._require_connected()
        raise NotImplementedError("Phase 3 stub — see connect().")

    async def get_drone_pose(
        self,
        drone_id: str,
    ) -> dict[str, float]:
        """Return the current pose of a drone from PX4 SITL.

        Returns
        -------
        {"lat": float, "lon": float, "alt_m": float,
         "roll": float, "pitch": float, "yaw": float}

        NotImplementedError — Phase 3 stub.
        """
        self._require_connected()
        raise NotImplementedError("Phase 3 stub — see connect().")

    async def send_velocity_command(
        self,
        drone_id: str,
        vx: float,
        vy: float,
        vz: float,
    ) -> None:
        """Send a velocity command to a drone via MAVROS.

        NotImplementedError — Phase 3 stub.
        """
        self._require_connected()
        raise NotImplementedError("Phase 3 stub — see connect().")

    async def get_all_poses(self) -> dict[str, dict[str, float]]:
        """Return poses for all active drones.

        NotImplementedError — Phase 3 stub.
        """
        self._require_connected()
        raise NotImplementedError("Phase 3 stub — see connect().")

    async def disconnect(self) -> None:
        """Shut down ROS2 node and release resources."""
        self._connected = False
        self._node = None

    # -----------------------------------------------------------------------
    # Drone ID ↔ PX4 instance mapping
    # -----------------------------------------------------------------------

    @staticmethod
    def drone_id_to_instance(drone_id: str) -> int:
        """Map drone ID to PX4 SITL instance number (1-indexed).

        S01-S38 → 1-38
        C1-C2   → 39-40
        """
        if drone_id.startswith("S"):
            try:
                return int(drone_id[1:])
            except ValueError:
                raise GazeboBridgeError(f"Invalid scout drone ID: {drone_id}")
        elif drone_id.startswith("C"):
            try:
                n = int(drone_id[1:])
                return 38 + n  # C1→39, C2→40
            except ValueError:
                raise GazeboBridgeError(f"Invalid commander drone ID: {drone_id}")
        raise GazeboBridgeError(f"Unrecognised drone ID format: {drone_id}")

    @staticmethod
    def instance_to_drone_id(instance: int, n_scouts: int = 38) -> str:
        """Map PX4 instance number back to drone ID."""
        if 1 <= instance <= n_scouts:
            return f"S{instance:02d}"
        elif instance > n_scouts:
            return f"C{instance - n_scouts}"
        raise GazeboBridgeError(f"Invalid instance number: {instance}")

    @staticmethod
    def camera_topic(drone_id: str, channel: str = "rgb", namespace: str = "/seapatch") -> str:
        """Return the ROS2 topic for a drone's camera.

        Scout RGB:      /seapatch/S22/camera/rgb/image_raw
        Commander RGB:  /seapatch/C1/camera/rgb/image_raw
        Commander FLIR: /seapatch/C1/camera/thermal/image_raw
        """
        if channel == "thermal" and not drone_id.startswith("C"):
            raise GazeboBridgeError(
                f"Thermal channel not available on scout drone {drone_id}. "
                "Only commander drones (C*) carry thermal cameras."
            )
        return f"{namespace}/{drone_id}/camera/{channel}/image_raw"

    def _require_connected(self) -> None:
        if not self._connected:
            raise GazeboBridgeError(
                "GazeboBridge is not connected. Call await bridge.connect() first. "
                "Note: GazeboBridge is a Phase 3 component. For Phase 1/2, "
                "use SeaDronesSeebridge instead."
            )
