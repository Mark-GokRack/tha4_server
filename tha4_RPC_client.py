import os
import sys
import pathlib
sys.path.append(str( 
    pathlib.Path(os.path.dirname( __file__)).joinpath("talking-head-anime-4-demo/src/")
))
sys.path.append( os.getcwd() )

from multiprocessing.managers import BaseManager
from concurrent.futures import ThreadPoolExecutor
import numpy as np

import time
import torch
import torchvision

from typing import Tuple, Optional
import argparse

from collections import deque

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import wx
from wx import glcanvas

from scipy.spatial.transform import Rotation

import cv2
import mediapipe

from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker

from tha4.mocap.mediapipe_constants import HEAD_ROTATIONS, HEAD_X, HEAD_Y, HEAD_Z
from tha4.mocap.mediapipe_face_pose import MediaPipeFacePose
from tha4.mocap.mediapipe_face_pose_converter_00 import MediaPoseFacePoseConverter00, MediaPipeFacePoseConverter00Args

class AverageCalculator:
    def __init__(self, count=100):
        self.count = count
        self.values = np.zeros( [self.count] )
        self.is_first_loop = True
        self.idx = 0
    def add(self, value):
        self.values[self.idx] = value
        self.idx+=1
        if self.idx >= self.count:
            self.is_first_loop = False
            self.idx = 0
    def get_average(self):
        if self.is_first_loop:
            if self.idx == 0:
                return 0.0
            else:
                return np.average( self.values[:self.idx] )
        else:
            return np.average( self.values )

class ImageGLCanvas(glcanvas.GLCanvas):
    def __init__(self, parent, width, height):
        self.size = wx.Size( width, height )
        super().__init__(parent, -1, size=self.size)
        self.init = False
        self.context = glcanvas.GLContext(self)
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.image_data = None
        self.image_texture = None

    def InitGL(self):
        glViewport(0, 0, self.size.width, self.size.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        self.image_texture = glGenTextures(1)
        
    def setImage( self, image_data ):
        self.image_data = image_data

    def OnPaint(self, event):
        # dc = wx.PaintDC(self)
        self.SetCurrent(self.context)
        if not self.init:
            self.InitGL()
            self.init = True
        self.OnDraw() 

    def OnDraw(self):
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if self.image_data is not None:
            glEnable( GL_TEXTURE_2D )
            glBindTexture(GL_TEXTURE_2D, self.image_texture )
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D( 
                GL_TEXTURE_2D, 0, GL_RGB,
                self.size.width, self.size.height,
                0, GL_RGB, GL_UNSIGNED_BYTE,
                self.image_data
            )
            glBegin(GL_QUADS)
            glTexCoord2d(0.0, 1.0)
            glVertex3d(-1.0, -1.0,  0.0)
            glTexCoord2d(1.0, 1.0)
            glVertex3d( 1.0, -1.0,  0.0)
            glTexCoord2d(1.0, 0.0)
            glVertex3d( 1.0,  1.0,  0.0)
            glTexCoord2d(0.0, 0.0)
            glVertex3d(-1.0,  1.0,  0.0)
            glEnd()
            glDisable( GL_TEXTURE_2D )
            # glFlush()
        self.SwapBuffers()

class MainFrame(wx.Frame):
    def __init__(
        self, host_ip : str, port : int,
        pose_converter: MediaPoseFacePoseConverter00,
        video_capture : cv2.VideoCapture,
        face_landmarker: FaceLandmarker,
        show_webcam_info : bool = False,
        delay_time : float = 0.0
    ):
        super().__init__(None, wx.ID_ANY, "THA4 Client : mediapipe puppeteer")
        self.face_landmarker = face_landmarker
        self.video_capture = video_capture
        self.pose_converter = pose_converter
        self.mediapipe_face_pose = None
        self.show_webcam_info = show_webcam_info
        self.delay_deque = deque()
        self.delay_time = delay_time

        BaseManager.register("tha4_scripts")
        self.manager = BaseManager(address=(host_ip, port), authkey=b"tha4_rpc")
        self.manager.connect()
        self.tha4_server = self.manager.tha4_scripts()
        self.executer = ThreadPoolExecutor(max_workers=1)
        self.last_update_time = None

        self.resp_time_calculator = AverageCalculator(100)

        self.create_ui(show_webcam_info)
        self.create_timer()

    def create_ui( self, show_webcam_info ):
        self.main_sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.main_sizer)
        self.SetAutoLayout(1)
        self.create_media_panel(self)
        self.main_sizer.Add(self.media_panel, wx.SizerFlags(0).Expand().Border(wx.ALL, 5))

        if show_webcam_info:
            self.create_capture_panel(self)
            self.main_sizer.Add( self.capture_panel, wx.SizerFlags(0).Expand().Border(wx.ALL, 5))

        self.main_sizer.Fit(self)

    def create_timer ( self ):
        self.result_timer = wx.Timer(self, wx.ID_ANY)
        self.Bind(wx.EVT_TIMER, self.update_result_panel, id=self.result_timer.GetId())
        # self.capture_timer = wx.Timer(self, wx.ID_ANY)
        # self.Bind(wx.EVT_TIMER, self.update_capture_panel, id=self.capture_timer.GetId())
        self.Bind(wx.EVT_CLOSE, self.on_close)

    def create_media_panel( self, parent ):
        self.media_panel = wx.Panel( parent, style=wx.SIMPLE_BORDER )
        self.media_panel_sizer = wx.BoxSizer( wx.HORIZONTAL )
        self.media_panel.SetSizer(self.media_panel_sizer)
        self.media_panel.SetAutoLayout(1)

        def current_pose_supplier() -> Optional[MediaPipeFacePose]:
            return self.mediapipe_face_pose
        self.pose_converter.init_pose_converter_panel(self.media_panel, current_pose_supplier)

        self.result_panel = wx.Panel( self.media_panel, size=(512,600), style=wx.SIMPLE_BORDER)
        self.result_panel_sizer = wx.BoxSizer( wx.VERTICAL )
        self.result_panel.SetSizer(self.result_panel_sizer)
        self.result_panel.SetAutoLayout(1)
        self.result_canvas = ImageGLCanvas( self.result_panel, 512, 512 )
        self.result_panel_sizer.Add( self.result_canvas, 0, wx.FIXED_MINSIZE )
        self.resp_time_text = wx.StaticText( self.result_panel, label="" )
        self.result_panel_sizer.Add( self.resp_time_text, wx.SizerFlags().Border() )
        self.media_panel_sizer.Add( self.result_panel, 0, wx.FIXED_MINSIZE )
        
        self.result_panel_sizer.Fit(self.result_panel)
        self.media_panel_sizer.Fit(self.media_panel)

    def create_capture_panel(self, parent):
        self.capture_panel = wx.Panel(parent, style=wx.RAISED_BORDER)
        self.capture_panel_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.capture_panel.SetSizer(self.capture_panel_sizer)
        self.capture_panel.SetAutoLayout(1)

        self.webcam_capture_panel = wx.Panel(self.capture_panel, size=(256, 192), style=wx.SIMPLE_BORDER)
        self.webcam_capture_panel_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.webcam_canvas = ImageGLCanvas( self.webcam_capture_panel, 256, 192 )
        self.webcam_capture_panel_sizer.Add( self.webcam_canvas, 0, wx.FIXED_MINSIZE )
        self.capture_panel_sizer.Add(self.webcam_capture_panel, wx.SizerFlags(0).FixedMinSize().Border(wx.ALL, 5))

        self.rotation_labels = {}
        self.rotation_value_labels = {}
        rotation_column = self.create_rotation_column(self.capture_panel, HEAD_ROTATIONS)
        self.capture_panel_sizer.Add(rotation_column, wx.SizerFlags(0).Expand().Border(wx.ALL, 3))

    def create_rotation_column(self, parent, rotation_names):
        column_panel = wx.Panel(parent, style=wx.SIMPLE_BORDER)
        column_panel_sizer = wx.FlexGridSizer(cols=2)
        column_panel_sizer.AddGrowableCol(1)
        column_panel.SetSizer(column_panel_sizer)
        column_panel.SetAutoLayout(1)

        for rotation_name in rotation_names:
            self.rotation_labels[rotation_name] = wx.StaticText(
                column_panel, label=rotation_name, style=wx.ALIGN_RIGHT)
            column_panel_sizer.Add(self.rotation_labels[rotation_name],
                                   wx.SizerFlags(1).Expand().Border(wx.ALL, 3))

            self.rotation_value_labels[rotation_name] = wx.TextCtrl(
                column_panel, style=wx.TE_RIGHT)
            self.rotation_value_labels[rotation_name].SetValue("0.00")
            self.rotation_value_labels[rotation_name].Disable()
            column_panel_sizer.Add(self.rotation_value_labels[rotation_name],
                                   wx.SizerFlags(1).Expand().Border(wx.ALL, 3))

        column_panel.GetSizer().Fit(column_panel)
        return column_panel

    def on_close(self, event: wx.Event):
        # Stop the timers
        self.result_timer.Stop()
        # self.capture_timer.Stop()

        self.executer.shutdown()

        # Destroy the windows
        self.Destroy()
        event.Skip()

    def update_result_panel(self, event:Optional[wx.Event] = None):
        time_now = time.perf_counter()

        if self.mediapipe_face_pose is not None:
            current_pose = np.asarray(
                self.pose_converter.convert(self.mediapipe_face_pose),
                dtype=np.float32
            )
        else:
            current_pose = np.zeros( [45], dtype=np.float32 )

        future_result = self.executer.submit( 
            self.tha4_server.get_png_from_pose,
            current_pose
        )
        
        self.update_capture_panel( event )

        png_image = future_result.result()

        image_data = torchvision.io.decode_png(
            torch.from_numpy( np.copy( np.frombuffer( png_image, dtype=np.uint8 ) ) )
        ).permute(1,2,0).numpy().tobytes()
        self.delay_deque.append( ( time_now + self.delay_time, image_data ))

        if len( self.delay_deque ) > 0 and self.delay_deque[0][0] <= time_now:
            _, image_data = self.delay_deque.popleft()
            self.result_canvas.setImage( image_data )
            # self.result_canvas.OnPaint( None )
            self.result_panel.Refresh()
            self.result_panel.Update()

        if self.last_update_time is not None:
            elapsed_time = time_now - self.last_update_time
            self.resp_time_calculator.add( elapsed_time * 1000.0 )
        self.last_update_time = time_now
        self.resp_time_text.SetLabelText( 
            "average response time = {0:.2f} ms".format( self.resp_time_calculator.get_average() )
        )
        # self.Refresh()
        # self.Update()

    def update_capture_panel(self, event: wx.Event):
        there_is_frame, frame = self.video_capture.read()
        if there_is_frame:
            rgb_frame = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1)
            if self.show_webcam_info:
                self.webcam_canvas.setImage( cv2.resize(rgb_frame, (256, 192)).tobytes() )
                self.webcam_capture_panel.Refresh()
                self.webcam_capture_panel.Update()
            time_ms = int(time.perf_counter() * 1000)
            mediapipe_image = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=rgb_frame)
            detection_result = self.face_landmarker.detect_for_video(mediapipe_image, time_ms)
            self.update_mediapipe_face_pose(detection_result)

    def update_mediapipe_face_pose(self, detection_result):
        if len(detection_result.facial_transformation_matrixes) == 0:
            return

        xform_matrix = detection_result.facial_transformation_matrixes[0]
        blendshape_params = {}
        for item in detection_result.face_blendshapes[0]:
            blendshape_params[item.category_name] = item.score
        M = xform_matrix[0:3, 0:3]
        rot = Rotation.from_matrix(M)
        euler_angles = rot.as_euler('xyz', degrees=True)

        if self.show_webcam_info:
            self.rotation_value_labels[HEAD_X].SetValue("%0.2f" % euler_angles[0])
            self.rotation_value_labels[HEAD_X].Refresh()
            self.rotation_value_labels[HEAD_Y].SetValue("%0.2f" % euler_angles[1])
            self.rotation_value_labels[HEAD_Y].Refresh()
            self.rotation_value_labels[HEAD_Z].SetValue("%0.2f" % euler_angles[2])
            self.rotation_value_labels[HEAD_Z].Refresh()

        self.mediapipe_face_pose = MediaPipeFacePose(blendshape_params, xform_matrix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GUI version of simple client app for tha4 server testing.')
    parser.add_argument(
        "-i", "--host_ip", action="store", 
        dest="host_ip",
        help="Hostname or IP address of tha4 server.",
        default="localhost",
        required = False
    )
    parser.add_argument(
        "-p", "--port", action="store", 
        dest="port",
        help="Port number of tha4 server.",
        default=9999, type=int, required = False
    )
    parser.add_argument(
        "-w", "--show_webcam", action="store_true", 
        dest="webcam",
        help="set flag to display webcam capture screen"
    )
    parser.add_argument(
        "-d", "--delay", action="store", 
        dest="delay",
        help="set image delaytime in second.",
        default=0.0, type=float, required = False
    )
    args = parser.parse_args()


    pose_args = MediaPipeFacePoseConverter00Args(
        jaw_open_min = 0.1,
        jaw_open_max = 0.3,
        eye_surprised_max= 0.4
    )
    pose_converter = MediaPoseFacePoseConverter00( pose_args )
    face_landmarker_base_options = mediapipe.tasks.BaseOptions(
        model_asset_path='talking-head-anime-4-demo/data/thirdparty/mediapipe/face_landmarker_v2_with_blendshapes.task')
    options = mediapipe.tasks.vision.FaceLandmarkerOptions(
        base_options=face_landmarker_base_options,
        running_mode=mediapipe.tasks.vision.RunningMode.VIDEO,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1)
    face_landmarker = FaceLandmarker.create_from_options(options)

    video_capture = cv2.VideoCapture(0)

    app = wx.App()
    main_frame = MainFrame(
        args.host_ip, args.port,
        pose_converter, video_capture, face_landmarker,
        args.webcam, args.delay
    )
    main_frame.Show(True)
    main_frame.result_timer.Start(1)
    app.MainLoop()
