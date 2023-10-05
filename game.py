import asyncio
from wlf import GazeData, GazeReceiver
import mido

notes = [20, 40, 60, 80]

async def main():
    midi_port = mido.open_output(mido.get_output_names()[0])
    receiver = GazeReceiver()
    while True:
        data = await receiver.recv()
        index = data.column_counts.index(max(data.column_counts))
        print(index)
        msg = mido.Message('note_on', note=notes[index], channel=index)
        midi_port.send(msg)
    
if __name__ == '__main__':
    asyncio.run(main())