import socket
import numpy as np

HOST, PORT = "localhost", 9999

import time
import tqdm

num_loop = 100
elapsed_time = 0.0
failed_counter = 0
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.connect((HOST, PORT))
    for _ in tqdm.tqdm(range(num_loop)):
        start_time = time.perf_counter()
        current_pose = np.random.rand( 45 ).astype( np.float32 ) * 0.8
        sock.sendall( current_pose.tobytes() )
        recv_data = sock.recv( 1024 * 1024 )
        if len( recv_data) > 1:
            png_data = recv_data
        else:
            failed_counter +=1
        end_time = time.perf_counter()
        elapsed_time += end_time - start_time
    
print( "fps = {0:f}".format( num_loop / elapsed_time ) )
print( "packet loss ratio : {0:d}".format( failed_counter ) )
with open( "recieved.png" , "wb" )  as fp:
    fp.write( png_data )
