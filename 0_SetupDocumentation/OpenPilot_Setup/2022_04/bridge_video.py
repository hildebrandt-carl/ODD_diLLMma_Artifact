#!/usr/bin/env python3
import argparse
import cv2
import math
import threading
import time
import os
from multiprocessing import Process, Queue
from typing import Any
from lib.keyboard_ctrl import keyboard_poll_thread
import h5py

import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
from lib.can import can_function

import cereal.messaging as messaging
from cereal import log
from cereal.visionipc.visionipc_pyx import VisionIpcServer, VisionStreamType  # pylint: disable=no-name-in-module, import-error
from common.basedir import BASEDIR
from common.numpy_fast import clip
from common.params import Params
from common.realtime import DT_DMON, Ratekeeper
from selfdrive.car.honda.values import CruiseButtons
from selfdrive.test.helpers import set_params_enabled

parser = argparse.ArgumentParser(description='Bridge between CARLA and openpilot.')
parser.add_argument('--filename', type=str, help="Should be provided relative to the home directory", default='')

args = parser.parse_args()

W, H = 1928, 1208
REPEAT_COUNTER = 5
PRINT_DECIMATION = 100
STEER_RATIO = 15.

global end_of_video
end_of_video = False

pm = messaging.PubMaster(['roadCameraState', 'sensorEvents', 'can', "gpsLocationExternal"])
sm = messaging.SubMaster(['carControl', 'controlsState'])


class VehicleState:
  def __init__(self):
    self.speed = 0
    self.angle = 0
    self.bearing_deg = 0.0
    self.vel = [0,0,0]
    self.cruise_button = 0
    self.is_engaged = False
    self.ignition = True


def steer_rate_limit(old, new):
  # Rate limiting to 0.5 degrees per step
  limit = 0.5
  if new > old + limit:
    return old + limit
  elif new < old - limit:
    return old - limit
  else:
    return new


class Camerad:
  def __init__(self):
    self.frame_id = 0
    self.vipc_server = VisionIpcServer("camerad")

    # TODO: remove RGB buffers once the last RGB vipc subscriber is removed
    self.vipc_server.create_buffers(VisionStreamType.VISION_STREAM_RGB_ROAD, 4, True, W, H)
    self.vipc_server.create_buffers(VisionStreamType.VISION_STREAM_ROAD, 40, False, W, H)
    self.vipc_server.start_listener()

    # set up for pyopencl rgb to yuv conversion
    self.ctx = cl.create_some_context()
    self.queue = cl.CommandQueue(self.ctx)
    cl_arg = f" -DHEIGHT={H} -DWIDTH={W} -DRGB_STRIDE={W*3} -DUV_WIDTH={W // 2} -DUV_HEIGHT={H // 2} -DRGB_SIZE={W * H} -DCL_DEBUG "

    # TODO: move rgb_to_yuv.cl to local dir once the frame stream camera is removed
    kernel_fn = os.path.join(BASEDIR, "selfdrive", "camerad", "transforms", "rgb_to_yuv.cl")
    prg = cl.Program(self.ctx, open(kernel_fn).read()).build(cl_arg)
    self.krnl = prg.rgb_to_yuv
    self.Wdiv4 = W // 4 if (W % 4 == 0) else (W + (4 - W % 4)) // 4
    self.Hdiv4 = H // 4 if (H % 4 == 0) else (H + (4 - H % 4)) // 4

  def cam_callback(self, image):
    img = np.frombuffer(image, dtype=np.dtype("uint8"))

    # convert RGB frame to YUV
    rgb = np.reshape(img, (H, W * 3))
    rgb_cl = cl_array.to_device(self.queue, rgb)
    yuv_cl = cl_array.empty_like(rgb_cl)
    self.krnl(self.queue, (np.int32(self.Wdiv4), np.int32(self.Hdiv4)), None, rgb_cl.data, yuv_cl.data).wait()
    yuv = np.resize(yuv_cl.get(), np.int32(rgb.size / 2))
    eof = int(self.frame_id * 0.05 * 1e9)

    # TODO: remove RGB send once the last RGB vipc subscriber is removed
    self.vipc_server.send(VisionStreamType.VISION_STREAM_RGB_ROAD, img.tobytes(), self.frame_id, eof, eof)
    self.vipc_server.send(VisionStreamType.VISION_STREAM_ROAD, yuv.data.tobytes(), self.frame_id, eof, eof)

    dat = messaging.new_message('roadCameraState')
    dat.roadCameraState = {
      "frameId": self.frame_id,
      "transform": [1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0]
    }
    pm.send('roadCameraState', dat)
    self.frame_id += 1


def imu_callback(imu, vehicle_state):
  vehicle_state.bearing_deg = math.degrees(imu.compass)
  dat = messaging.new_message('sensorEvents', 2)
  dat.sensorEvents[0].sensor = 4
  dat.sensorEvents[0].type = 0x10
  dat.sensorEvents[0].init('acceleration')
  dat.sensorEvents[0].acceleration.v = [0, 0, 0]
  # copied these numbers from locationd
  dat.sensorEvents[1].sensor = 5
  dat.sensorEvents[1].type = 0x10
  dat.sensorEvents[1].init('gyroUncalibrated')
  dat.sensorEvents[1].gyroUncalibrated.v = [0, 0, 0]
  pm.send('sensorEvents', dat)


def panda_state_function(vs: VehicleState, exit_event: threading.Event):
  global end_of_video
  pm = messaging.PubMaster(['pandaStates'])
  while (not exit_event.is_set()) and (end_of_video != True):
    dat = messaging.new_message('pandaStates', 1)
    dat.valid = True
    dat.pandaStates[0] = {
      'ignitionLine': vs.ignition,
      'pandaType': "blackPanda",
      'controlsAllowed': True,
      'safetyModel': 'hondaNidec'
    }
    pm.send('pandaStates', dat)
    time.sleep(0.5)


def peripheral_state_function(exit_event: threading.Event):
  global end_of_video
  pm = messaging.PubMaster(['peripheralState'])
  while (not exit_event.is_set()) and (end_of_video != True):
    dat = messaging.new_message('peripheralState')
    dat.valid = True
    # fake peripheral state data
    dat.peripheralState = {
      'pandaType': log.PandaState.PandaType.blackPanda,
      'voltage': 12000,
      'current': 5678,
      'fanSpeedRpm': 1000
    }
    pm.send('peripheralState', dat)
    time.sleep(0.5)


def gps_callback(gps, vehicle_state):
  dat = messaging.new_message('gpsLocationExternal')

  # transform vel from carla to NED
  # north is -Y in CARLA
  velNED = [
    -vehicle_state.vel.y,  # north/south component of NED is negative when moving south
    vehicle_state.vel.x,  # positive when moving east, which is x in carla
    vehicle_state.vel.z,
  ]

  dat.gpsLocationExternal = {
    "timestamp": int(time.time() * 1000),
    "flags": 1,  # valid fix
    "accuracy": 1.0,
    "verticalAccuracy": 1.0,
    "speedAccuracy": 0.1,
    "bearingAccuracyDeg": 0.1,
    "vNED": velNED,
    "bearingDeg": vehicle_state.bearing_deg,
    "latitude": gps.latitude,
    "longitude": gps.longitude,
    "altitude": gps.altitude,
    "speed": vehicle_state.speed,
    "source": log.GpsLocationData.SensorSource.ublox,
  }

  pm.send('gpsLocationExternal', dat)


def fake_driver_monitoring(exit_event: threading.Event):
  global end_of_video
  pm = messaging.PubMaster(['driverState', 'driverMonitoringState'])
  while (not exit_event.is_set()) and (end_of_video != True):
    # dmonitoringmodeld output
    dat = messaging.new_message('driverState')
    dat.driverState.faceProb = 1.0
    pm.send('driverState', dat)

    # dmonitoringd output
    dat = messaging.new_message('driverMonitoringState')
    dat.driverMonitoringState = {
      "faceDetected": True,
      "isDistracted": False,
      "awarenessStatus": 1.,
    }
    pm.send('driverMonitoringState', dat)

    time.sleep(DT_DMON)


def can_function_runner(vs: VehicleState, exit_event: threading.Event):
  global end_of_video
  i = 0
  while (not exit_event.is_set()) and (end_of_video != True):
    can_function(pm, vs.speed, vs.angle, i, vs.cruise_button, vs.is_engaged)
    time.sleep(0.01)
    i += 1


def read_video_file(filename: str, camerad: Camerad, exit_event: threading.Event):
  global end_of_video
  rk = Ratekeeper(20)
  # Load the video
  cap = cv2.VideoCapture(filename)
  while not exit_event.is_set():
    ret, frame = cap.read()
    if not ret:
      end_of_video = True
      exit_event.set()
      break
    frame = cv2.resize(frame, (W, H)) 
    camerad.cam_callback(frame)
    rk.keep_time()

def bridge(q, filename):
  global end_of_video

  vehicle_state = VehicleState()
  camerad = Camerad()

  # launch fake car threads
  threads = []
  exit_event = threading.Event()
  threads.append(threading.Thread(target=panda_state_function, args=(vehicle_state, exit_event,)))
  threads.append(threading.Thread(target=peripheral_state_function, args=(exit_event,)))
  threads.append(threading.Thread(target=fake_driver_monitoring, args=(exit_event,)))
  threads.append(threading.Thread(target=can_function_runner, args=(vehicle_state, exit_event,)))
  threads.append(threading.Thread(target=read_video_file, args=(filename, camerad, exit_event,)))
  for t in threads:
    t.start()

  # can loop
  rk = Ratekeeper(100, print_delay_threshold=0.05)

  # init
  throttle_ease_out_counter = REPEAT_COUNTER
  brake_ease_out_counter = REPEAT_COUNTER
  steer_ease_out_counter = REPEAT_COUNTER

  is_openpilot_engaged = True
  throttle_out = steer_out = brake_out = 0
  throttle_op = steer_op = brake_op = 0
  throttle_manual = steer_manual = brake_manual = 0

  old_steer = old_brake = old_throttle = 0
  throttle_manual_multiplier = 0.7  # keyboard signal is always 1
  brake_manual_multiplier = 0.7  # keyboard signal is always 1
  steer_manual_multiplier = 45 * STEER_RATIO  # keyboard signal is always 1

  out_filename = filename[:filename.rfind(".")] + ".h5"
  h5_out = h5py.File(out_filename, "w")
  write_counter = 0

  while (not exit_event.is_set()) and (end_of_video == False):
    # 1. Read the throttle, steer and brake from op or manual controls
    # 2. Set instructions in Carla
    # 3. Send current carstate to op via can

    cruise_button = 0
    throttle_out = steer_out = brake_out = 0.0
    throttle_op = steer_op = brake_op = 0
    throttle_manual = steer_manual = brake_manual = 0.0

    # Update the sub master
    sm.update(0)

    # Get the steering angle
    steer_op = sm['carControl'].actuators.steeringAngleDeg

    # Get the alerts
    alert1 = sm["controlsState"].alertText1
    alert2 = sm["controlsState"].alertText2

    # Define a compound data type
    dtype = np.dtype([
        ('steering_angle', np.float32),
        ('frame_id', np.int64),
        ('alert1', h5py.string_dtype(encoding='utf-8')),
        ('alert2', h5py.string_dtype(encoding='utf-8'))
    ])

    # Create a structured array with the data
    data = np.array([(steer_op, camerad.frame_id, alert1, alert2)], dtype=dtype)

    # Create the dataset with a suitable key
    h5_out.create_dataset("{0:09d}_data".format(write_counter), data=data)

    # Increment the write counter
    write_counter += 1

    # --------------Step 2-------------------------------
    steer_out = steer_op 
    old_steer = steer_out

    # --------------Step 3-------------------------------
    vel = [0,0,0]
    speed = 50/3.6  # in m/s
    vehicle_state.speed = speed
    vehicle_state.vel = vel
    vehicle_state.angle = steer_out
    vehicle_state.cruise_button = cruise_button
    vehicle_state.is_engaged = is_openpilot_engaged

    if rk.frame % PRINT_DECIMATION == 0:
      print("frame: ", "engaged:", is_openpilot_engaged)

    rk.keep_time()

  # Clean up resources in the opposite order they were created.
  exit_event.set()
  for t in reversed(threads):
    t.join()

  h5_out.close()


def bridge_keep_alive(q: Any, filename: str):
  while 1:
    try:
      bridge(q, filename)
      break
    except RuntimeError:
      print("Restarting bridge...")


if __name__ == "__main__":

  # make sure params are in a good state
  set_params_enabled()

  if len(args.filename) == 0:
    print("Please provide a filename")
    exit()
  else:
    home_path = os.path.expanduser('~')
    fname = home_path + args.filename
    print("Processing: {}".format(fname))

  msg = messaging.new_message('liveCalibration')
  msg.liveCalibration.validBlocks = 20
  msg.liveCalibration.rpyCalib = [0.0, 0.0, 0.0]
  Params().put("CalibrationParams", msg.to_bytes())

  q: Any = Queue()
  p = Process(target=bridge_keep_alive, args=(q, fname), daemon=True)
  p.start()

  p.join()
