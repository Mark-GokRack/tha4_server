# tha4_server

this scripts aim to implement server-client model to [talking-head-anime-4-demo](https://github.com/pkhungurn/talking-head-anime-4-demo).


## Instructions

### prepare mediapipe

- at first, you have to download [face_landmarker_v2_with_blendshapes.task](https://github.com/nlml/deconstruct-mediapipe/blob/main/face_landmarker_v2_with_blendshapes.task) and place in following folder.
  ```path
  ./talking-head-anime-4-demo/data/thirdparty/mediapipe/
  ```

### prepare python environment.

- Python version should be 3.10.11 due to the talking-head-anime-4-demo.
  - It's useful to use [pyenv](https://github.com/pyenv/pyenv) is convenient, so if you have no religious concerns, it's worth considering.

- installing scripts
  - for windows
    ```cmd
    python -m venv .venv --prompt tha4_srv
    .venv\Scripts\activate.bat
    python.exe -m pip install --upgrade pip
    pip install -r requrements.txt
    ```
  - for linux
    ```bash
    python -m venv .venv --prompt tha4_srv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -r requrements.txt
    ```

## start server

- start server application with following command:
  ```
  python tha4_server.py
  ```
  - usage can show with "-h" option.
    ```
    > python tha4_server.py -h
    usage: tha4_server.py [-h] [-m MODEL_PATH] [-i HOST_IP] [-p PORT]

    start tha4 server.

    options:
    -h, --help            show this help message and exit
    -m MODEL_PATH, --model_path MODEL_PATH
                        file path of the character model (*.yaml)
    -i HOST_IP, --host_ip HOST_IP
                        Hostname or IP address for socket listening.
    -p PORT, --port PORT  Port number for socket listening.
    ```

  - Set the IP address to the address of the PC that will run this server script.


## start client

- start client application with following command:
  ```
  python tha4_client.py
  ```
  - usage can show with "-h" option.
    ```
    > python tha4_client.py -h
    usage: tha4_client.py [-h] [-i HOST_IP] [-p PORT] [-w]

    GUI version of simple client app for tha4 server testing.

    options:
      -h, --help            show this help message and exit
      -i HOST_IP, --host_ip HOST_IP
                            Hostname or IP address of tha4 server.
      -p PORT, --port PORT  Port number of tha4 server.
      -w, --show_webcam     set flag to display webcam capture screen
      -d DELAY, --delay DELAY
                            set image delaytime in second.
    ```
  
    - Set the IP address to the address of the PC that is running above server script.

