import os
import argparse
from zipfile import ZipFile
from tqdm import tqdm
import imageio
import load_defected_data

'''
Splits in to train,validation,test and respects original format, and clears erroneous scenes
for windows users, run git bash as admin and call python from there
'''
TRAIN_SCENE = ['scene_%05d' % i for i in range(0, 3000)]
VALID_SCENE = ['scene_%05d' % i for i in range(3000, 3250)]
TEST_SCENE = ['scene_%05d' % i for i in range(3250, 3500)]

parser = argparse.ArgumentParser()
parser.add_argument('--in_root', required=True)
parser.add_argument('--out_train_root', default='data/st3d_train_full_raw_light')
parser.add_argument('--out_valid_root', default='data/st3d_valid_full_raw_light')
parser.add_argument('--out_test_root', default='data/st3d_test_full_raw_light')
parser.add_argument('--defected_path', required=True)
args = parser.parse_args()

def prepare_dataset(scene_ids, out_dir):
    defected, defected_scenes, defected_rooms = load_defected_data.parseDefectedData(args.defected_path)
    for scene_id in tqdm(scene_ids):
        if scene_id in defected_scenes:
            continue
        scene_path_in = os.path.join(args.in_root, scene_id)
        rooms_path = os.path.join(scene_path_in, "2D_rendering")
        rooms_path_out = os.path.join(out_dir, scene_id, "2D_rendering")
        os.makedirs(rooms_path_out, exist_ok = True)
        os.symlink(os.path.join(scene_path_in, "annotation_3d.json"), os.path.join(os.path.dirname(rooms_path_out), "annotation_3d.json"))
        for room in os.listdir(rooms_path):
            if room in defected_rooms:
                continue
            room_out = os.path.join(rooms_path_out, room, "panorama")
            room_in = os.path.join(rooms_path, room, "panorama")
            os.makedirs(room_out, exist_ok = True)
            os.symlink(os.path.join(room_in, "camera_xyz.txt"), os.path.join(room_out, "camera_xyz.txt"))
            os.symlink(os.path.join(room_in, "layout.txt"), os.path.join(room_out, "layout.txt"))

            for room_type in ["empty", "full", "simple"]:
                room_type_out = os.path.join(room_out, room_type)
                room_type_in = os.path.join(room_in, room_type)
                os.makedirs(room_type_out, exist_ok = True)
                for datum in os.listdir(room_type_in):
                    os.symlink(os.path.join(room_type_in, datum), os.path.join(room_type_out, datum))


if __name__ == "__main__":
    prepare_dataset(TRAIN_SCENE, args.out_train_root)
    prepare_dataset(VALID_SCENE, args.out_valid_root)
    prepare_dataset(TEST_SCENE, args.out_test_root)


