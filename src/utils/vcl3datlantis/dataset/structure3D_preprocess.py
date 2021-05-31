import os
import sys
import argparse
from load_defected_data import parseDefectedData
'''
    IF THERE IS SUMLINK PRIVILEDGE ERROR, RUN GIT BASH AS ADMIN AND EXECUTE FROM THERE
'''

def parseArguments(args):
    usage_text = (
        "Structure 3D dataset transformer"
        "Usage:  python main.py [options],"
        "   with [options]:"
    )
    parser = argparse.ArgumentParser(description=usage_text)

    parser.add_argument('--input_path', type=str,required=True, help='Path to the dataset.')
    parser.add_argument('--defected_scenes', type=str, default = None, help='Path to the defects text file.')
    parser.add_argument('--output_path', type=str,required=True, help='Path to save the symlinks.')

    return parser.parse_known_args(args)

if __name__ == "__main__":
    args, _ = parseArguments(sys.argv)

    if not os.path.exists(args.input_path):
        raise RuntimeError(args.input_path + " does not exist.")

    os.makedirs(args.output_path, exist_ok=True)

    defected, defected_scenes, defected_rooms = parseDefectedData(args.defeceted_scenes)

    scenes = os.listdir(args.input_path)
    i = 0
    for scene in scenes:
        i += 1
        print("Progress : {} of {}".format(i,len(scenes)), end = '\r')
        if scene in defected_scenes:
            print("Skip " + scene)
            continue
        full_scene_path = os.path.join(args.input_path, scene, "2D_rendering")
        for room in os.listdir(full_scene_path):
            whole_name = scene + "_room_" + room
            if whole_name in defected_rooms:
                print("Skip " + whole_name)
                continue

            full_room_path = os.path.join(full_scene_path, room, "panorama")
            for room_type in [os.path.basename(x[0]) for x in os.walk(full_room_path)][1:]:
                full_root_type_path = os.path.join(full_room_path, room_type)
                rgb_images = [x for x in os.listdir(full_root_type_path) if "rgb" in x]
                other_data = [x for x in os.listdir(full_root_type_path) if "rgb" not in x]
                for light_type_fname in [x for x in rgb_images]:
                    light_type = light_type_fname.split('.')[0].split('_')[1]
                    sample_name = scene + "_room_" + room + "_" + room_type + "_" + light_type
                    output_name = os.path.join(args.output_path,sample_name)
                    os.makedirs(output_name, exist_ok=True)

                    input = os.path.join(full_root_type_path, light_type_fname)
                    output = os.path.join(output_name, light_type_fname)
                    os.symlink(input, output)

                    input = os.path.join(full_room_path, "layout.txt")
                    output = os.path.join(output_name, "layout.txt")
                    os.symlink(input, output)
                    
                    for rest_data in other_data:
                        input = os.path.join(full_root_type_path, rest_data)
                        output = os.path.join(output_name, rest_data)
                        os.symlink(input, output)
