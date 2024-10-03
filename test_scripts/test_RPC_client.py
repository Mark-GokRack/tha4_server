import numpy as np

from multiprocessing.managers import BaseManager
import time
import tqdm

from typing import Tuple
import argparse

import torch
import torchvision

def proc( host_ip : str, port_num : int ) -> None:
    num_loop = 1000
    elapsed_time = 0.0
    failed_counter = 0

    BaseManager.register("tha4_scripts")
    manager = BaseManager(address=(host_ip, port_num), authkey=b"tha4_rpc")
    manager.connect()
    tha4_server = manager.tha4_scripts()

    for _ in tqdm.tqdm(range(num_loop)):
        start_time = time.perf_counter()
        current_pose = np.random.rand( 45 ).astype( np.float32 ) * 0.4
        png_data = tha4_server.get_png_from_pose( current_pose )
        tensor_png_data = torch.from_numpy( np.frombuffer( png_data, dtype=np.uint8 ) )
        image_data = torchvision.io.decode_png(
            tensor_png_data
        ).permute(1,2,0).numpy()
        #else:
        #    image_data = np.frombuffer( frame_buffer, dtype=np.uint8 )
        #    png_data = None
        end_time = time.perf_counter()
        elapsed_time += end_time - start_time
        
    print( "average fps : {0:f}".format( ( num_loop - failed_counter ) / elapsed_time ) )
    with open( "recieved.png" , "wb" )  as fp:
        fp.write( png_data )

    np.save("recieved.npy", image_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simple client app for tha4 server testing.')
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
    args = parser.parse_args()
    proc( args.host_ip, args.port )