import os
import sys
import pathlib

import numpy as np
import zlib
import pickle

import torch
import torchvision

from multiprocessing.managers import BaseManager
import argparse

sys.path.append(str( 
    pathlib.Path(os.path.dirname( __file__)).joinpath("talking-head-anime-4-demo/src/")
))
sys.path.append( os.getcwd() )

import time
from typing import Optional

from scipy.spatial.transform import Rotation

import mediapipe
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker

from tha4.mocap.mediapipe_face_pose import MediaPipeFacePose

from tha4.charmodel.character_model import CharacterModel
from tha4.image_util import convert_linear_to_srgb

class Tha4Scripts:
    def __init__( self, model_path : str ):
        self.device = torch.device("cuda:0")
        self.character_model = CharacterModel.load( model_path )
        self.source_image = self.character_model.get_character_image( self.device )
        self.poser = self.character_model.get_poser( self.device )
        self.background_image = torch.ones( 4, 512, 512, device=self.device )
        self.png_compress_level  = 1

        """
        face_landmarker_base_options = mediapipe.tasks.BaseOptions(
            model_asset_path='talking-head-anime-4-demo/data/thirdparty/mediapipe/face_landmarker_v2_with_blendshapes.task')
        options = mediapipe.tasks.vision.FaceLandmarkerOptions(
            base_options=face_landmarker_base_options,
            running_mode=mediapipe.tasks.vision.RunningMode.VIDEO,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1)
        self.face_landmarker = FaceLandmarker.create_from_options(options)
        """

    def __blend_with_background( self, numpy_image, background ):
        alpha = numpy_image[3:4, :, :]
        color = numpy_image[0:3, :, :]
        new_color = color * alpha + (1.0 - alpha) * background[0:3, :, :]
        return torch.cat([new_color, background[3:4, :, :]], dim=0)

    def __convert_pose_to_image( self, current_pose : np.ndarray ) -> torch.Tensor:
        pose = torch.tensor( current_pose, dtype=self.poser.get_dtype(), device=self.device )
        output_image = self.poser.pose( self.source_image, pose )[0].float()
        output_image = torch.clip((output_image + 1.0) / 2.0, 0.0, 1.0)
        output_image = self.__blend_with_background( output_image, self.background_image )
        output_image = 255.0 * convert_linear_to_srgb(output_image)
        output_image = output_image.byte().detach().cpu()[:3,:,:]
        return output_image 
    
    def get_image_from_pose( self, current_pose : np.ndarray ) -> bytearray:
        output_image = self.__convert_pose_to_image( current_pose )
        output_image = output_image.numpy()
        return output_image
    
    def get_png_from_pose( self, current_pose : np.ndarray ) -> bytearray:
        output_image = self.__convert_pose_to_image(current_pose)
        png_data = torchvision.io.encode_png(
            output_image,
            self.png_compress_level
        ).numpy()
        return png_data

    def get_zipped_image_from_pose( self, current_pose : np.ndarray ) -> bytearray:
        output_image = self.__convert_pose_to_image( current_pose )
        output_image = output_image.numpy()
        zipped_image = zlib.compress( pickle.dumps(output_image), level=1 )
        return zipped_image

    """
    def pose_convert( self, mediapipe_face_pose, pose_args )->np.ndarray:
        return np.asarray( np.zeros( [45] ), dtype=np.float32 )

    def get_png_from_camera_image( self, png_image : bytearray, pose_args, time_ms : Optional[int] = None ) -> bytearray:

        rgb_frame = torchvision.io.decode_png(
            image_data = torchvision.io.decode_png(
                torch.from_numpy( np.frombuffer( png_image, dtype=np.uint8 ) )
            ).permute(1,2,0).numpy().tobytes()
        )
        if time_ms is None:
            time_ms = int(time.perf_counter() * 1000)

        mediapipe_image = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.face_landmarker.detect_for_video(mediapipe_image, time_ms)

        xform_matrix = detection_result.facial_transformation_matrixes[0]
        blendshape_params = {}
        for item in detection_result.face_blendshapes[0]:
            blendshape_params[item.category_name] = item.score

        mediapipe_face_pose = MediaPipeFacePose(blendshape_params, xform_matrix)

        return self.get_png_from_pose( pose_convert( mediapipe_face_pose, pose_args ) )
    """

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='start tha4 server.')
    parser.add_argument(
        "-m", "--model_path", action="store", 
        dest="model_path",
        help="file path of the character model (*.yaml)",
        default="talking-head-anime-4-demo/data/character_models/lambda_00/character_model.yaml",
        required = False
    )
    parser.add_argument(
        "-i", "--host_ip", action="store", 
        dest="host_ip",
        help="Hostname or IP address for socket listening.",
        default="",
        required = False
    )
    parser.add_argument(
        "-p", "--port", action="store", 
        dest="port",
        help="Port number for socket listening.",
        default=9999, type=int, required = False
    )
    args = parser.parse_args()    

    converter = Tha4Scripts( args.model_path )

    BaseManager.register( "tha4_scripts", lambda : converter )
    manager = BaseManager(address=(args.host_ip, args.port), authkey=b"tha4_rpc" )
    server = manager.get_server()
    server.serve_forever()
