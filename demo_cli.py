#!/usr/bin/env python3

from dataclasses import dataclass
import pickle
from pathlib import Path
import cv2
from xmlrpc.client import ServerProxy
from pkm.real.cseg.fit import SegmenterApp
from pkm.real.multi_rs_camera import MultiRSCamera, CameraConfig


@dataclass
class Config:
    host: str = 'localhost'
    port: int = 8001


def main(cfg: Config):
    from xmlrpc.client import ServerProxy
    predictor = ServerProxy(F'http://{cfg.host}:{cfg.port}')
    # vid_path: str = './example_data/test.mp4'
    # vid_path: str = '/input/WHAM/examples/20241217_224650.mp4'
    # vid_path: str = '/tmp/docker/20250111_150030.mp4'
    # vid_path: str = '/tmp/docker/sav6/color.mp4'
    vid_path: str = '/tmp/docker/sav10/out.mp4'
    cam_path: str = '/tmp/docker/sav10/cam.pkl'

    with open(cam_path, 'rb') as fp:
        fx = pickle.load(fp)['K'][0, 0]

    out_path: str = '/tmp/docker/hamer_test/out_hand3.pkl'
    Path(out_path).parent.mkdir(parents=True,
                                exist_ok=True)
    out = predictor.hand(vid_path,
                         out_path,
                         float(fx))
    with open(out_path, 'rb') as fp:
        data = pickle.load(fp)
    print(len(data))
    for d in data:
        print(d)


if __name__ == '__main__':
    main(Config())
