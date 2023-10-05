import asyncio
import pickle
import socket
from typing import Callable
from .gaze_data import GazeData

class GazeReceiver:
    def __init__(self, port = 49988):
        # See https://gist.github.com/ninedraft/7c47282f8b53ac015c1e326fffb664b5
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        #self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self._socket.bind(("", port))
        
    async def recv(self) -> GazeData:
        loop = asyncio.get_event_loop()
        data = await loop.sock_recv(self._socket, 1500)
        return pickle.loads(data)
    
    async def close(self):
        self._socket.close()

async def main():
    receiver = GazeReceiver()
    while True:
        print(await receiver.recv())
    
if __name__ == '__main__':
    asyncio.run(main())