import time
import redis
import math

from .gaze_data import GazeData, Face, Point2DF, Point3DF


class GazeSender:
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client

    def send(self, data: GazeData):
        json_data = data.model_dump_json()
        self.redis_client.publish('gaze', json_data)


def main():
    client = redis.Redis()
    sender = GazeSender(redis_client=client)

    faces = [
        Face(camera_centroid_norm=Point2DF(x=0.1, y=0.5), gaze_vector=Point3DF(
            x=0.0, y=0.0, z=0.0), gaze_screen_intersection_norm=Point2DF(x=0.1, y=0.5)),
        Face(camera_centroid_norm=Point2DF(x=0.8, y=0.5), gaze_vector=Point3DF(
            x=0.0, y=0.0, z=0.0), gaze_screen_intersection_norm=Point2DF(x=0.5, y=0.5)),
    ]

    try:
        while True:
            for face in faces:
                face.gaze_screen_intersection_norm.x = math.fmod(
                    face.gaze_screen_intersection_norm.x + 0.1, 1)

            sender.send(GazeData(faces=faces))
            time.sleep(0.1)
    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
