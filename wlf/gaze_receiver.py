import asyncio
import pickle
import socket
import struct
from typing import Callable
from .gaze_data import GazeData

class GazeReceiver:
    def __init__(self, mcast_grp="224.0.0.224", port=49988):
        self._socket = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self._socket.bind(('', port))
        mreq = struct.pack("4sl", socket.inet_aton(mcast_grp), socket.INADDR_ANY)
        self._socket.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        
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