import pickle
import time
from .gaze_data import GazeData
import socket
import struct

class GazeSender:
    def __init__(self, mcast_grp="224.0.0.224", port=49988):
        self._port = port
        self._mcast_grp = mcast_grp
        
        ttl = struct.pack('b', 1)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)
        self._socket.setblocking(False)

    def send(self, data: GazeData):
        pickled = pickle.dumps(data)
        assert len(pickled) < 1500
        self._socket.sendto(pickled, (self._mcast_grp, self._port))

    def close(self):
        self._socket.close()


def main():
    sender = GazeSender()

    try:
        i = 0
        while True:
            i = (i + 1) % 100
            sender.send(
                GazeData(column_counts=[i, (i + 25) % 100, (i + 50) % 100, (i + 75) % 100]))
            time.sleep(0.01)
    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        pass

    sender.close()


if __name__ == '__main__':
    main()
