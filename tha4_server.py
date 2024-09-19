import os
import sys
import pathlib
sys.path.append(str( 
    pathlib.Path(os.path.dirname( __file__)).joinpath("talking-head-anime-4-demo/src/")
))
sys.path.append( os.getcwd() )

import numpy as np
import time
import tqdm

from tha4.charmodel.character_model import CharacterModel
from tha4.image_util import convert_linear_to_srgb

import torch
import torchvision

import socketserver

def blend_with_background( numpy_image, background):
    alpha = numpy_image[3:4, :, :]
    color = numpy_image[0:3, :, :]
    new_color = color * alpha + (1.0 - alpha) * background[0:3, :, :]
    return torch.cat([new_color, background[3:4, :, :]], dim=0)

def get_tha4_handler( model_path : str ) -> socketserver.BaseRequestHandler:
    class Tha4Handler( socketserver.BaseRequestHandler ):
        def setup(self):
            self.device = torch.device("cuda:0")
            self.character_model = CharacterModel.load( model_path )
            self.source_image = self.character_model.get_character_image( self.device )
            self.poser = self.character_model.get_poser( self.device )
            self.background_image = torch.ones( 4, 512, 512, device=self.device )
            self.png_compress_level  = 1
            
        def handle(self):
            while data := self.request.recv(1024):
                try:
                    current_pose = np.frombuffer( data.strip(), dtype=np.float32 )
                except:
                    self.request.sendall( b'\x00' )
                else:
                    if len(current_pose) != 45:
                        self.request.sendall( b'\x00' )
                    else:
                        pose = torch.tensor( current_pose, dtype=self.poser.get_dtype(), device=self.device )
                        output_image = self.poser.pose( self.source_image, pose )[0].float()
                        output_image = torch.clip((output_image + 1.0) / 2.0, 0.0, 1.0)
                        output_image = blend_with_background( output_image, self.background_image )
                        output_image = 255.0 * convert_linear_to_srgb(output_image)
                        output_image = output_image.byte()
                        png_data = torchvision.io.encode_png(
                            output_image.detach().cpu()[:3,:,:],
                            self.png_compress_level
                        ).numpy().tobytes()
                        send_data = bytearray()
                        send_length = len(png_data)
                        header = bytes( ("{0}:".format( send_length )).encode() )
                        send_data += header
                        send_data += png_data

                        self.request.sendall( send_data )
    return Tha4Handler

def start( model_path : str, host_ip : str, num_port : int ) -> None:
    with socketserver.TCPServer((host_ip, num_port), get_tha4_handler(model_path)) as server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.server_close()

if __name__ == "__main__":
    if len( sys.argv ) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "talking-head-anime-4-demo/data/character_models/lambda_00/character_model.yaml"
    if len( sys.argv ) > 2:
        host_ip = sys.argv[2]
    else:
        host_ip = "localhost"
        #host_ip = "192.168.10.16"
    if len( sys.argv ) > 3:
        num_port = int( sys.argv[3] )
    else:
        num_port = 9999
    
    start( model_path, host_ip, num_port )
