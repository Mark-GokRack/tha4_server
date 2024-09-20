import socket
import numpy as np

import time
import tqdm

from typing import Tuple
import argparse

import torch
import torchvision

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
            length_str, ignored, frame_buffer = frame_buffer.partition(b":")
            length = int( length_str )
    return recv_failed, frame_buffer

def proc( host_ip : str, port_num : int ) -> None:
    num_loop = 100
    elapsed_time = 0.0
    failed_counter = 0
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((host_ip, port_num))
        for _ in tqdm.tqdm(range(num_loop)):
            start_time = time.perf_counter()
            current_pose = np.random.rand( 45 ).astype( np.float32 ) * 0.4
            sock.sendall( current_pose.tobytes() )
            recv_failed, frame_buffer = recv_all( sock )
            if not recv_failed:
                png_data = frame_buffer

                tensor_png_data = torch.from_numpy( np.frombuffer( png_data, dtype=np.uint8 ) )
                image_data = torchvision.io.decode_png(
                    tensor_png_data
                ).permute(1,2,0).numpy()

                end_time = time.perf_counter()
                elapsed_time += end_time - start_time
            else:
                failed_counter +=1
        
    print( "average fps : {0:f}".format( ( num_loop - failed_counter ) / elapsed_time ) )
    print( "packet loss ratio : {0:.2f} %".format( 100 *  failed_counter / num_loop ) )
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