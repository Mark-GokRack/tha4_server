import os
import sys
import pathlib
sys.path.append(str( 
    pathlib.Path(os.path.dirname( __file__)).joinpath("../talking-head-anime-4-demo/src/")
))
sys.path.append( os.getcwd() )

import io
import numpy as np
import time
import tqdm
from PIL import Image

from scipy.spatial.transform import Rotation

from tha4.charmodel.character_model import CharacterModel
from tha4.image_util import convert_linear_to_srgb

import torch
import torchvision

def proc(model_path : str ) -> None:
    device = torch.device("cuda:0")
    character_model = CharacterModel.load( model_path )
    source_image = character_model.get_character_image( device )
    poser = character_model.get_poser( device )
    current_pose = np.zeros( [45], dtype=np.float32 ) 
    elapsed_time = 0.0
    num_loop = 1000
    png_compress_level  = 1
    png_data = None
    for _ in tqdm.tqdm( range( num_loop ) ):
        start_time = time.perf_counter()
        pose = torch.tensor( current_pose, dtype=poser.get_dtype(), device=device )
        output_image = poser.pose( source_image, pose )[0].float()
        output_image = torch.clip((output_image + 1.0) / 2.0, 0.0, 1.0)
        output_image = convert_linear_to_srgb(output_image)
        c, h, w = output_image.shape
        output_image = 255.0 * torch.transpose(output_image.reshape(c, h * w), 0, 1).reshape(h, w, c)
        output_image = output_image.byte()
        numpy_image = output_image.detach().cpu().numpy()
        if True:
            with io.BytesIO() as png_image:
                Image.fromarray( numpy_image ).save(
                    png_image, format="PNG", optimize=False, compress_level=png_compress_level
                )
                png_data = png_image.getvalue()
        end_time = time.perf_counter()
        elapsed_time += end_time - start_time
    print( "fps = {0:f}".format( num_loop / elapsed_time ) )

    with open( "result.png", "wb" ) as fp:
        if png_data is not None:
            fp.write( png_data )
        else:
            Image.fromarray( numpy_image ).save(
                fp, format="PNG", optimize=False, compress_level=png_compress_level
            )

if __name__ == "__main__":
    if len( sys.argv ) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "talking-head-anime-4-demo/data/character_models/lambda_00/character_model.yaml"
    proc( model_path )