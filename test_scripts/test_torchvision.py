import os
import sys
import pathlib
sys.path.append(str( 
    pathlib.Path(os.path.dirname( __file__)).joinpath("../talking-head-anime-4-demo/src/")
))
sys.path.append( os.getcwd() )

import numpy as np
import time
import tqdm

from tha4.charmodel.character_model import CharacterModel
from tha4.image_util import convert_linear_to_srgb

import torch
import torchvision

def blend_with_background( numpy_image, background):
    alpha = numpy_image[3:4, :, :]
    color = numpy_image[0:3, :, :]
    new_color = color * alpha + (1.0 - alpha) * background[0:3, :, :]
    return torch.cat([new_color, background[3:4, :, :]], dim=0)

def proc(model_path : str ) -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    character_model = CharacterModel.load( model_path )
    source_image = character_model.get_character_image( device )
    poser = character_model.get_poser( device )
    current_pose = np.zeros( [45], dtype=np.float32 ) 
    elapsed_time = 0.0
    num_loop = 1000
    png_compress_level  = 1
    png_data = None
    background_image = torch.ones( 4, 512, 512, device=device )
    for _ in tqdm.tqdm( range( num_loop ) ):
        start_time = time.perf_counter()
        pose = torch.tensor( current_pose, dtype=poser.get_dtype(), device=device )
        output_image = poser.pose( source_image, pose )[0].float()
        output_image = torch.clip((output_image + 1.0) / 2.0, 0.0, 1.0)
        output_image = blend_with_background( output_image, background_image )
        output_image = 255.0 * convert_linear_to_srgb(output_image)
        output_image = output_image.byte()
        png_data = torchvision.io.encode_png(
             output_image.detach().cpu()[:3,:,:], png_compress_level
        ).numpy().tobytes()
        end_time = time.perf_counter()
        elapsed_time += end_time - start_time
    print( "fps = {0:f}".format( num_loop / elapsed_time ) )

    with open( "result.png", "wb" ) as fp:
        fp.write( png_data )

if __name__ == "__main__":
    if len( sys.argv ) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "talking-head-anime-4-demo/data/character_models/lambda_00/character_model.yaml"
    proc( model_path )