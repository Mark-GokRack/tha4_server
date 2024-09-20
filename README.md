# tha4_server

## Instructions

### prepare mediapipe

- download [face_landmarker_v2_with_blendshapes.task](https://github.com/nlml/deconstruct-mediapipe/blob/main/face_landmarker_v2_with_blendshapes.task) and place in following path.
  ```path
  ./talking-head-anime-4-demo/data/thirdparty/mediapipe/face_landmarker_v2_with_blendshapes.task
  ```

### prepare python
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


## start client

- start client application with following command:
  ```
  python tha4_client_character_model_mediapipe_puppeteer.py
  ```
  - usage can show with "-h" option.
    ```
    > python tha4_client_character_model_mediapipe_puppeteer.py -h
    usage: tha4_client_character_model_mediapipe_puppeteer.py [-h] [-i HOST_IP] [-p PORT]

    GUI version of simple client app for tha4 server testing.

    options:
      -h, --help            show this help message and exit
      -i HOST_IP, --host_ip HOST_IP
                            Hostname or IP address of tha4 server.
    -p PORT, --port PORT  Port number of tha4 server.
    ```

