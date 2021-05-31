def parseDefectedData(defected_scenes_path):
    defected = []
    defected_scenes = []
    defected_rooms = []
    if defected_scenes is not None:
        with open(defected_scenes_path) as f:
            defected = f.read().splitlines()
            defected_rooms = [x for x in defected if "_room_" in x]
            defected_scenes = [x for x in defected if "scene_" in x and x not in defected_rooms]
        print(str(len(defected_rooms) + len(defected_scenes)) + " defectes loaded")

    return defected, defected_scenes, defected_rooms