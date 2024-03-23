import os, sys, pdb
import numpy as np
import pickle
import shutil
import argparse
from utils import track_file_format_transfer, load_feat_from_pickle
from tracklet import Tracklet

scenarios = 'S003'
def argument_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--src_root', type=str, default=f"./data/track_results_{scenarios}/", 
            help='the root path of single camera tracking results')
    parser.add_argument('--dst_root', type=str, default="./data/preprocessed_data/", 
            help='the root path of preprocessed results')
    parser.add_argument('--feat_root', type=str, default=f"./data/track_feats_{scenarios}/", 
            help='the root path of features of each tracklet')
    return parser

def generate_all_preprocessed_track_info(cam_dict, dst_root):
    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)
    os.makedirs(dst_root)

    cam_list        = []
    track_list      = []
    in_out_info_obj = open(os.path.join(dst_root, 'in_out_all.txt'), 'w')
    for cam_id, tracklet_dict in cam_dict.items():
        for track_id, tracklet in tracklet_dict.items():
            assert (tracklet.cam_id == cam_id)
            cam_list.append(cam_id)
            track_list.append(track_id)
            in_out_info_obj.write('{} {} {} {}\n'.format(cam_id, track_id,
                tracklet.frames[0], tracklet.frames[-1])) # [cam_id, track_id, start_frame, end_frame]
    in_out_info_obj.close()
    cam_arr     = np.array(cam_list)
    track_arr   = np.array(track_list)
    np.save(os.path.join(dst_root, 'cam_vec.npy'), cam_arr)
    np.save(os.path.join(dst_root, 'track_vec.npy'), track_arr)
    



def lala():
    """ This is the latest version.
    Load the original pickle files for the purpose of constructing the feature array for final matching.
    It is saved to Pickle format. 
    """
    
    cam_list = os.listdir(f'./data/track_results_{scenarios}')
    print(cam_list)
    fr_dict = {}
    for cam in cam_list:
        with open(os.path.join(f'./data/track_results_{scenarios}', cam), 'r') as fid:
            for line in fid.readlines():
                s = [int(float(x)) for x in line.rstrip().split(' ')] # [frame_id, track_id, x, y, w, h, ...]
                cam_id      = int(cam[2:4])
                frame_id    = s[0]
                track_id    = s[1]
                if cam_id not in fr_dict:
                    fr_dict[cam_id] = {}
                    
            
                if track_id not in fr_dict[cam_id]:
                    fr_dict[cam_id][track_id] = []
                    
                fr_dict[cam_id][track_id].append(frame_id)
               
    return fr_dict


import os, sys, pdb
import numpy as np
import pickle
import random
from tqdm import tqdm


def is_valid_position(x, y, w, h):
    """
    To judge if the position is out of the image

    Args:
        None
    """
    x_max = 1917
    y_max = 1077
    x2 = x + w
    y2 = y + h
    if x < 0 or y < 0 or x2 >= x_max or y2 >= y_max:
        return False
    return True


def process_border(x, y, w, h):
    x_max = 1917
    y_max = 1077 

    dw, dh = 0, 0
    if x < 1:
        dw = -x
        x = 1
    if y < 1:
        dh = -y
        y = 1
    x2, y2 = x + w, y + h
    w = x_max - x if x2 >= x_max else w - dw
    h = y_max - y if y2 >= y_max else h - dh
    return (x, y, w, h)


def get_static_10_color():
    """ For visualization
    """
    color_list = [[]] * 10
    color_list[0] = (255, 0, 0)
    color_list[1] = (0, 255, 0)
    color_list[2] = (0, 0, 255)
    color_list[3] = (255, 255, 0)
    color_list[4] = (255, 0, 255)
    color_list[5] = (0, 255, 255)
    color_list[6] = (128, 0, 0)
    color_list[7] = (0, 128, 0)
    color_list[8] = (0, 0, 128)
    color_list[9] = (138, 43, 226)
    return color_list


def get_color(num_color=365):
    """
    For visualization

    Args:
        num_color: The number of colors will be involved in visualization
    """
    num_ch = np.power(num_color, 1/3.)
    frag = int(np.floor(256/num_ch))
    color_ch = range(0, 255, frag)
    color_list = []
    for color_r in color_ch:
        for color_g in color_ch:
            for color_b in color_ch:
                color_list.append((color_r, color_g, color_b))
    random.shuffle(color_list)
    # color_list[0] = (255, 0, 0); color_list[1] = (0, 255, 0); color_list[2] = (0, 0, 255); color_list[3] = (255, 255, 0); color_list[4] = (138, 43, 226)
    return color_list

def track_file_format_transfer(src_root, dst_root):
    """ Transfer file format of track results.
    Single camera format (the direct output file of single camera algorithm) -> Multi camera format (the submission format)
    All files must be named as "c04x.txt"

    Args:
        src_root:
        dst_root:
    """
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)

    cam_list = os.listdir(src_root)
    cam_list.sort()
    for cam_file in cam_list:
        print ('processing: {}'.format(cam_file))
        cam_id = int(cam_file[1:4]) 
        dst_obj = open(os.path.join(dst_root, cam_file), 'w')
        f_dict = {}
        with open(os.path.join(src_root, cam_file), 'r') as fid:
            for line in fid.readlines():
                s = [int(float(x)) for x in line.rstrip().split(' ')] # [frame_id, track_id, x, y, w, h, ...]
                x, y, w, h = s[2:6]
                if not is_valid_position(x, y, w, h): # to drop those frames that are beyond of the image
                    x, y, w, h = process_border(x, y, w, h)
                    if w <= 0 or h <= 0:
                        continue
                    s[2:6] = x, y, w, h
                fr_id = s[0]
                line = '{} {} {} {} -1 -1\n'.format(cam_id, s[1], s[0], ' '.join(map(str, s[2:6]))) # [camera_id, track_id, frame_id, x, y, w, h, -1, -1]
                if fr_id not in f_dict:
                    f_dict[fr_id] = []
                f_dict[fr_id].append(line)

            fr_ids = sorted(f_dict.keys())
            for fr_id in fr_ids:
                for line in f_dict[fr_id]:
                    dst_obj.write(line)
        dst_obj.close()

def load_feat_from_pickle(src_root):
    """ This is the latest version.
    Load the original pickle files for the purpose of constructing the feature array for final matching.
    It is saved to Pickle format. 
    """
    feat_dict = {}
    fr_dict = {}
    src_list = sorted(os.listdir(src_root))
    # for src_file in tqdm(src_list):
    for src_file in src_list:
        src_path = os.path.join(src_root, src_file)
        f_dict = pickle.load(open(src_path, 'rb'), encoding='latin1')
        for k, v in f_dict.items():
            if v is None or v.size < 30 or np.all(v == 0): # v.size < 30: to prevent some bad features
                continue
            s = k.split('_')  # ['c001', '55', '9'] 
            track_id = int(s[-1]) # 9
            cam_id = int(s[0][1:]) # 1
            fr_id = int(s[1])  # 55 khi làm thật chỉnh lại cho đúng
            k = f'c0{cam_id}_{fr_id}_{track_id}'
            if cam_id not in feat_dict:
                feat_dict[cam_id] = {}
                fr_dict[cam_id] = {}
            if track_id not in feat_dict[cam_id]:
                feat_dict[cam_id][track_id] = []
                fr_dict[cam_id][track_id] = [] 
            feat_dict[cam_id][track_id].append(v)
            fr_dict[cam_id][track_id].append(fr_id)
    return feat_dict, fr_dict
    


def preprocess():
    parser = argument_parser()
    args = parser.parse_args()

    # tmp_root = os.path.join('tmp', args.src_root.split('/')[-1])
    tmp_root = './tmp'
    if os.path.exists(tmp_root):
        shutil.rmtree(tmp_root)
    os.makedirs(tmp_root)
    print ('* transfering the format of track files...')
    track_file_format_transfer(args.src_root, tmp_root)
    cam_list = os.listdir(tmp_root)
    cam_list.sort()
    cam_dict = {}
    for cam_file in cam_list:
        print('--------------------------------------')
        print()
        print ('processing: {}'.format(cam_file))
        cam_id = int(cam_file[1:4]) 
        tracklet_dict = {}
        with open(os.path.join(tmp_root, cam_file), 'r') as fid:
            for line in fid.readlines():
                s = [int(x) for x in line.rstrip().split()] # [cam_id, track_id, frame_id, x, y, w, h, -1, -1]
                c_id, track_id, fr_id, x, y, w, h = s[:-2]
                assert (c_id == cam_id)
                xc = (x + w / 2.)
                yc = (y + h / 2.)
                if s[1] not in tracklet_dict:
                    tracklet_dict[track_id] = Tracklet(c_id, xc, yc, fr_id)
                else:
                    tracklet_dict[track_id].add_element(xc, yc, fr_id)
        assert cam_id not in cam_dict
        cam_dict[cam_id] = tracklet_dict
    
    
    print ('* generating all preprocessed track information...')
    generate_all_preprocessed_track_info(cam_dict, args.dst_root)
    
    print ('* processing features...')
    fr_dict  = lala()
    feat_dict, fr_dict = load_feat_from_pickle(args.feat_root)
    with open(os.path.join(args.dst_root, 'feat_all_vec.pkl'), 'wb') as fid:
        pickle.dump(feat_dict, fid)
    with open(os.path.join(args.dst_root, 'frame_all_vec.pkl'), 'wb') as fid:
        pickle.dump(fr_dict, fid)

if __name__ == '__main__':
    preprocess()