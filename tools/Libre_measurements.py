import time
import numpy as np
from tools.Save_as_file import Save_to_file
import sys

from zaber_motion import Units
from zaber_motion.binary import Connection
from tools.libreVNA import libreVNA


def preform_sweep(freq_start, freq_stop,num_point = 500):
    # Create the control instance
    vna = libreVNA('localhost', 19542)

    # Quick connection check (should print "LibreVNA-GUI")
    print(vna.query("*IDN?"))

    # Make sure we are connecting to a device (just to be sure, with default settings the LibreVNA-GUI auto-connects)
    vna.cmd(":DEV:CONN")
    dev = vna.query(":DEV:CONN?")
    if dev == "Not connected":
        print("Not connected to any device, aborting")
        exit(-1)
    else:
        print("Connected to "+dev)

    # Capture live data as it is coming in. Stop acquisition for now
    vna.cmd(":VNA:ACQ:STOP")

    # switch to VNA mode, set up the sweep parameters
    print("Setting up the sweep...")
    vna.cmd(":DEV:MODE VNA")
    vna.cmd(":VNA:SWEEP FREQUENCY")
    vna.cmd(":VNA:STIM:LVL -10")
    vna.cmd(":VNA:ACQ:IFBW 100")
    vna.cmd(":VNA:ACQ:AVG 1")
    vna.cmd(f":VNA:ACQ:POINTS {num_point + 1}")
    vna.cmd(f":VNA:FREQuency:START {freq_start * 1e9}")
    vna.cmd(f":VNA:FREQuency:STOP {freq_stop * 1e9}")
    
    sweepComplete = [False]

    data_arr = []


    def callback(data):

        percent = data['pointNum']/num_point

        filled = int(40 * percent)

        bar = '█' * filled + '-' * (40-filled)

        sys.stdout.write(f'\r{'Measuring '}|{bar}|{percent*100:.1f}%')
        sys.stdout.flush()

        

        data_arr.append(data)

        if data["pointNum"] == num_point:
            # this was the last point
            vna.remove_live_callback(19000, callback)
            sweepComplete[0] = True        

    # Set up the connection for the live data
    vna.add_live_callback(19000, callback)
    print("Starting the sweep...")
    vna.cmd(":VNA:ACQ:RUN")

    while not sweepComplete[0]:
       time.sleep(0.1)

    print("Sweep complete")

    return data_arr


def libre_vna_measurement(measurement_scheme, filename, start_freq = 5, end_freq =6, home=False):

    con = Connection.open_serial_port('/dev/ttyUSB0')

    dev = con.detect_devices()
    
    if home:
        dev[0].home()
        print('Homed')

    distances = measurement_scheme.get_points()

    distances += np.abs(distances[0])

    data = []

    for i, dist in enumerate(distances):
        print(f'Preforming Measurement {i} of {len(distances)}')
        dev[0].move_absolute(dist, Units.LENGTH_MILLIMETRES)

        data.append(preform_sweep(start_freq, end_freq))

    file_saver = Save_to_file(f'{filename}.zip', 'Sweep_measurement', distances)

    file_saver.write_data_to_file(data)