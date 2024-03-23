"""
Date: 2022/04/19
"""

import os, sys, pdb
import numpy as np
import math

# for angle mode, it is deprecated in the latest version!
thres = [(0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (-1, -1)]

class Tracklet(object):
    """ This class is used to record all necessary information and calculate the trajectory of this tracklet.
    The tracklet keeps all information of each tracker. 
    
    Attributes:
        cam_id: camera id
        x, y: the central coordinates of this track in the special frame
        fr_id: frame id
    """

    def __init__(self, cam_id, x, y, fr_id):
        """ Initialize TrackletClass.
        """
        self.cam_id = cam_id
        self.co = [(x, y)]
        self.frames = [fr_id]
   
    def add_element(self, x, y, fr_id):
        """ Key function. 
        It adds information and calculate the trajectory.
        """
        self.co.append((x, y))
        self.frames.append(fr_id)


if __name__ == '__main__':
    src_root = './tmp/track_results/'
    dst_root = './data/preprocessed_data/'

    cam_list = os.listdir(src_root)
    cam_list.sort()
    for cam_file in cam_list:
        print ('processing: {}'.format(cam_file))
        cam_id = int(cam_file[1:4]) # c04x.txt -> 4x
        tracklet_dict = {}
        with open(os.path.join(src_root, cam_file), 'r') as fid:
            for line in fid.readlines():
                s = [int(x) for x in line.rstrip().split()] # [cam_id, track_id, frame_id, x, y, w, h, -1, -1]
                c_id, track_id, fr_id, x, y, w, h = s[:-2]
                assert (c_id == cam_id)
                xc = (x + w / 2.)
                yc = (y + h / 2.)
                if s[1] not in tracklet_dict:
                    tracklet_dict[track_id] = Tracklet(c_id, xc, yc, fr_id, -1, 4) # initialized theta=-1 and angle id=4
                else:
                    tracklet_dict[track_id].add_element(xc, yc, fr_id)
