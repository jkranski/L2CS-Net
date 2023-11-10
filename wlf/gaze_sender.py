import time
import redis

from .gaze_data import GazeData


class GazeSender:
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client

    def send(self, data: GazeData):
        json_data = data.model_dump_json()
        self.redis_client.publish('gaze', json_data)


def main():
    client = redis.Redis()
    sender = GazeSender(redis_client=client)

    try:
        i = 0
        while True:
            i = (i + 1) % 100
            sender.send(
                GazeData(column_counts=[i, (i + 25) % 100, (i + 50) % 100, (i + 75) % 100]))
            time.sleep(0.1)
    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
