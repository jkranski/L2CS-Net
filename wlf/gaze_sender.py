import pickle
import time
from .gaze_data import GazeData
import redis


class GazeSender:
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client

    def send(self, data: GazeData):
        pickled = pickle.dumps(data)
        assert len(pickled) < 1500
        self.redis_client.publish('gaze', pickled)


def main():
    client = redis.Redis()
    sender = GazeSender(redis_client=client)

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
