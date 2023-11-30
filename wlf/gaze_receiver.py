import pickle
from typing import Callable
from gaze_data import GazeData
import redis


class GazeReceiver:
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.channel = self.redis_client.pubsub()
        self.channel.subscribe("gaze")

    def recv(self, timeout) -> GazeData:
        msg = self.channel.get_message(
            ignore_subscribe_messages=True, timeout=timeout)
        if msg is not None:
            return msg['data']


def main():
    client = redis.Redis()
    receiver = GazeReceiver(client)
    while True:
        print(receiver.recv(timeout=1))


if __name__ == '__main__':
    main()
