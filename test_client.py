import socket
import numpy as np
import sys

import time
import tqdm

def proc( host_ip : str, port_num : int ):
    num_loop = 100
    elapsed_time = 0.0
    failed_counter = 0
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((host_ip, port_num))
        for _ in tqdm.tqdm(range(num_loop)):
            start_time = time.perf_counter()
            current_pose = np.random.rand( 45 ).astype( np.float32 ) * 0.8
            sock.sendall( current_pose.tobytes() )
            recv_data = sock.recv( 1024 * 1024 )
            if len( recv_data ) > 1:
                png_data = recv_data
            else:
                failed_counter +=1
            end_time = time.perf_counter()
            elapsed_time += end_time - start_time
        
    print( "fps = {0:f}".format( num_loop / elapsed_time ) )
    print( "packet loss ratio : {0:d}".format( failed_counter ) )
    with open( "recieved.png" , "wb" )  as fp:
        fp.write( png_data )


if __name__ == "__main__":
    if len( sys.argv ) > 1:
        host_ip = sys.argv[1]
    else:
        host_ip = "localhost"
    if len( sys.argv ) > 2:
        num_port = int( sys.argv[2] )
    else:
        num_port = 9999
    proc( host_ip, num_port )