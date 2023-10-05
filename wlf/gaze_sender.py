import pickle
import websockets.sync.client
import time
from .gaze_data import GazeData
import socket

class GazeSender:
    def __init__(self, port = 49988):
        self._port = port
        # See https://gist.github.com/ninedraft/7c47282f8b53ac015c1e326fffb664b5
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        #self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self._socket.setblocking(False)
    
    def send(self, data: GazeData):
        pickled = pickle.dumps(data)
        assert len(pickled) < 1500
        self._socket.sendto(pickled, ('<broadcast>', self._port))
        
    def close(self):
        self._socket.close()

def main():
    sender = GazeSender()
    
    try:
        while True:
            sender.send(GazeData(column_counts=[0, 0, 1, 0]))
            time.sleep(1)
    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        pass
        
    sender.close()
    
if __name__ == '__main__':
    main()