import os
import pickle
def storePaths(root_path, output_path):
    paths = [dirpath for dirpath, dirnames, _ in os.walk(root_path) if not dirnames]
    print("saving {} paths".format(len(paths)))
    with open(output_path, 'wb') as fp:
        pickle.dump(paths, fp)

def loadPaths(filename):
    with open (filename, 'rb') as fp:
        itemlist = pickle.load(fp)
    return itemlist
if __name__ == "__main__":
    p = "Path"
    out = r"Path"

    storePaths(p,out)

    print("loaded {} paths".format(len(loadPaths(out))))
