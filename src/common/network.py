import socket
import json
import logging
import threading
from typing import Callable, Dict, Any, Tuple, Optional

# Set up logging format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UDPSender:
    """
    Sends 3D coordinates and tracking metadata from the processing PC to the Raspberry Pi.
    Uses UDP protocol for low latency, as outdated coordinates are irrelevant for real-time interception.
    """
    def __init__(self, ip: str, port: int) -> None:
        self.ip: str = ip
        self.port: int = port
        self.socket: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        logger.info(f"UDPSender initialized. Target: {self.ip}:{self.port}")

    def send_target_position(self, x: float, y: float, z: float, detected: bool, timestamp: float, 
                             pred_x: float = 0.0, pred_y: float = 0.0, t_impact: float = 0.0) -> None:
        """
        Sends the 3D position of the ball and metadata as a compact JSON string over UDP.
        
        :param x: X-coordinate (horizontal position along the goal line, e.g. in mm)
        :param y: Y-coordinate (height, e.g. in mm)
        :param z: Z-coordinate (distance from goal, e.g. in mm)
        :param detected: True if the ball is actively tracked, False if tracking is lost
        :param timestamp: Unix timestamp when the camera frame was captured
        """
        data: Dict[str, Any] = {
            "x": round(x, 2),
            "y": round(y, 2),
            "z": round(z, 2),
            "det": int(detected),
            "t": timestamp,
            "px": round(pred_x, 2),
            "py": round(pred_y, 2),
            "ti": round(t_impact, 3)
        }
        
        try:
            message: bytes = json.dumps(data).encode('utf-8')
            self.socket.sendto(message, (self.ip, self.port))
        except Exception as e:
            logger.error(f"Failed to send UDP packet: {e}")

    def close(self) -> None:
        """Closes the network socket."""
        self.socket.close()
        logger.info("UDPSender socket closed.")


class UDPReceiver:
    """
    Listens for 3D coordinates on the Raspberry Pi 5.
    Can be run synchronously or in a background thread with a callback function.
    """
    def __init__(self, ip: str, port: int) -> None:
        self.ip: str = ip
        self.port: int = port
        self.socket: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Allow reuse of local addresses
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Bind to port
        try:
            self.socket.bind((self.ip, self.port))
            logger.info(f"UDPReceiver bound to {self.ip}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to bind UDPReceiver socket: {e}")
            raise e
            
        self.running: bool = False
        self.thread: Optional[threading.Thread] = None

    def receive_packet(self) -> Optional[Dict[str, Any]]:
        """
        Blocks until a packet is received, then parses and returns it.
        
        :return: Parsed data dictionary or None if error occurs.
        """
        try:
            data, addr = self.socket.recvfrom(1024)
            parsed_data: Dict[str, Any] = json.loads(data.decode('utf-8'))
            return parsed_data
        except socket.timeout:
            return None
        except Exception as e:
            if self.running:
                logger.error(f"Error receiving/parsing packet: {e}")
            return None

    def start_listening(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Starts listening for packets in a background thread.
        Calls the provided callback function with the parsed data dictionary.
        
        :param callback: Function to process the received coordinates.
        """
        if self.running:
            logger.warning("UDPReceiver is already listening.")
            return

        self.running = True
        self.socket.settimeout(1.0)  # Avoid permanent blocking when shutting down
        self.thread = threading.Thread(target=self._listener_loop, args=(callback,), daemon=True)
        self.thread.start()
        logger.info("UDPReceiver listener thread started.")

    def _listener_loop(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Internal loop running in the background thread."""
        while self.running:
            packet = self.receive_packet()
            if packet is not None:
                try:
                    callback(packet)
                except Exception as e:
                    logger.error(f"Error in UDPReceiver callback: {e}")

    def stop_listening(self) -> None:
        """Stops the background listener thread and closes the socket."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None
        self.socket.close()
        logger.info("UDPReceiver stopped and socket closed.")
