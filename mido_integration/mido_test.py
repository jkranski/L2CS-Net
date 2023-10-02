import mido
import time
msg = mido.Message('note_on', note=60)
port = mido.open_output(mido.get_output_names()[0])

port.send(msg)
time.sleep(5)  # Necessary to keep the script from exiting prior to hearing the sound

