import sys, signal
sys.path.insert(0, "build/lib.linux-armv7l-2.7/")

import VL53L1X
import time
from datetime import datetime
from voice import playvoice

tof = VL53L1X.VL53L1X(i2c_bus=1, i2c_address=0x29)
print("Python: Initialized")
tof.open()
print("Python: Opened")
tof.set_user_roi(VL53L1X.VL53L1xUserRoi(6, 9, 9, 6))
tof.start_ranging(3)

def exit_handler(signal, frame):
    tof.stop_ranging()
    tof.close()
    sys.exit(0)

signal.signal(signal.SIGINT, exit_handler)

while True:
    distance_cm = tof.get_distance()/10
    if distance_cm < 0:
        # Error -1185 may occur if you didn't stop ranging in a previous test
        print("Error: {}".format(distance_cm))
    else:
        print("Distance: {}cm".format(distance_cm))
    time.sleep(0.5)
    if distance_cm >=150 and distance_cm <=200:
        playvoice('1.5m.m4a')
