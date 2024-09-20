import socket
import numpy as np

import time
import torch
import torchvision

from typing import Tuple, Optional
import argparse

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import wx
from wx import glcanvas

def recv_all( sock : socket.socket ) -> Tuple[ bool, bytearray ]:
    length = None
    frame_buffer = bytearray()
    recv_failed = False
    while True:
        recv_data = sock.recv( 1024 )
        if length is None and recv_data == b"x00":
            recv_failed = True
            break
        if len( recv_data ) == 0:
            recv_failed = True
            break
        frame_buffer += recv_data
        if len( frame_buffer ) == length:
            recv_failed = False
            break
        if length is None:
            if b":" not in frame_buffer:
                recv_failed = True
                break
            length_str, _, frame_buffer = frame_buffer.partition(b":")
            length = int( length_str )
    return recv_failed, frame_buffer

class FpsStatistics:
    def __init__(self):
        self.count = 100
        self.fps = np.zeros( [self.count] )
        self.is_first_loop = True
        self.idx = 0
    def add_fps(self, fps):
        self.fps[self.idx] = fps
        self.idx+=1
        if self.idx >= self.count:
            self.is_first_loop = False
            self.idx = 0
    def get_average_fps(self):
        if self.is_first_loop:
            return np.average( self.fps[:self.idx] )
        else:
            return np.average( self.fps )

class ImageGLCanvas(glcanvas.GLCanvas):
    def __init__(self, parent, width, height):
        self.size = wx.Size( width, height )
        dispAttrs = wx.glcanvas.GLAttributes()
        # dispAttrs.PlatformDefaults().DoubleBuffer().Depth(32).EndList()
        glcanvas.GLCanvas.__init__(
            self, parent, -1, #dispAttrs
            size=self.size
        )
        self.init = False
        self.context = glcanvas.GLContext(self)
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        
        self.image_data = None
        self.image_texture = None

    def InitGL(self):
        glEnable(GL_TEXTURE_2D)
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
        dc = wx.PaintDC(self)
        self.SetCurrent(self.context)
        if not self.init:
            self.InitGL()
            self.init = True
        self.OnDraw() 

    def OnDraw(self):
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if self.image_data is not None:
            """
            glDrawPixels(
                self.size.width,
                self.size.height,
                GL_RGB, GL_UNSIGNED_BYTE,
                self.image_data.data
            )
            """
            glEnable( GL_TEXTURE_2D )
            glBindTexture(GL_TEXTURE_2D, self.image_texture )
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D( 
                GL_TEXTURE_2D, 0, GL_RGB,
                self.size.width, self.size.height,
                0, GL_RGB, GL_UNSIGNED_BYTE,
                self.image_data.data
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
            glFlush()
        self.SwapBuffers()

class MainFrame(wx.Frame):

    def __init__(self, host_ip : str, port : int ):
        super().__init__(None, wx.ID_ANY, "THA4 Client : random pose")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect( (host_ip, port) )
        self.last_update_time = None

        self.fps_statistics = FpsStatistics()

        self.create_ui()
        self.create_timer()

    def create_ui( self ):
        self.main_sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.main_sizer)
        self.SetAutoLayout(1)
        self.create_animation_panel(self)
        self.main_sizer.Add(self.image_panel, wx.SizerFlags(0).Expand().Border(wx.ALL, 5))
        self.main_sizer.Fit(self)

    def create_timer ( self ):
        self.socket_timer = wx.Timer(self, wx.ID_ANY)
        self.Bind(wx.EVT_TIMER, self.update_png_data, id=self.socket_timer.GetId())
        self.Bind(wx.EVT_CLOSE, self.on_close)

    def create_animation_panel( self, parent ):
        self.image_panel = wx.Panel( parent, size=(512,512), style=wx.SIMPLE_BORDER )
        self.image_panel_sizer = wx.BoxSizer()
        self.image_panel.SetSizer(self.image_panel_sizer)
        self.image_panel.SetAutoLayout(1)
        self.image_canvas_gl = ImageGLCanvas( self.image_panel, 512, 512 )
        self.image_canvas_gl_sizer = wx.BoxSizer(wx.VERTICAL)
        self.image_canvas_gl.SetSizer(self.image_canvas_gl_sizer)
        self.image_canvas_gl.SetAutoLayout(1)
        self.image_panel_sizer.Add( self.image_canvas_gl, 0, wx.FIXED_MINSIZE )
        self.image_canvas_gl_sizer.Fit(self.image_canvas_gl)
        self.image_panel_sizer.Fit(self.image_panel)


    def on_close(self, event: wx.Event):
        # Stop the timers
        self.socket_timer.Stop()
        # Stop the socket.
        self.sock.shutdown(socket.SHUT_RDWR)
        self.sock.close()
        # Destroy the windows
        self.Destroy()
        event.Skip()

    def update_png_data(self, event:Optional[wx.Event] = None):
        current_pose = np.random.rand( 45 ).astype( np.float32 ) * 0.4
        self.sock.sendall( current_pose.tobytes() )
        recv_failed, png_image = recv_all( self.sock )

        if not recv_failed:
            image_data = torchvision.io.decode_png(
                torch.from_numpy( np.frombuffer( png_image, dtype=np.uint8 ) )
            ).permute(1,2,0).numpy()
            self.image_canvas_gl.setImage( image_data )

        time_now = time.perf_counter()
        if self.last_update_time is not None:
            elapsed_time = time_now - self.last_update_time
            fps = 1.0 / elapsed_time
            self.fps_statistics.add_fps( fps )
        self.last_update_time = time_now

        self.Refresh()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simple client app for tha4 server testing.')
    parser.add_argument(
        "-i", "--host_ip", action="store", 
        dest="host_ip",
        help="Hostname or IP address of tha4 server.",
        # default="localhost",
        default="192.168.10.16",
        required = False
    )
    parser.add_argument(
        "-p", "--port", action="store", 
        dest="port",
        help="Port number of tha4 server.",
        default=9999, type=int, required = False
    )
    args = parser.parse_args()

    app = wx.App()
    main_frame = MainFrame(args.host_ip, args.port)
    main_frame.Show(True)
    main_frame.socket_timer.Start(1)
    app.MainLoop()
