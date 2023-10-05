import asyncio
from wlf import GazeData, GazeReceiver
import mido

notes = [20, 40, 60, 80]


async def main():
    midi_port = mido.open_output(mido.get_output_names()[0])
    receiver = GazeReceiver()
    while True:
        data = await receiver.recv()
        for i in range(0, 4):
            msg = mido.Message(
                'control_change', value=data.column_counts[i], channel=i)
            print(msg)
            midi_port.send(msg)

if __name__ == '__main__':
    asyncio.run(main())
