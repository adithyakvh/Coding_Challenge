from yolov5.detect import run
import os
import re

def number_split(s):
    return filter(None, re.split(r'(\d+)', s))

if __name__ == '__main__':
    # run(weights='yolov5l.pt', source=0, save_txt=True, classes=32)
    all_results_folder = os.path.join('yolov5', 'runs', 'detect')

    # Check if folder 'exp' exists and rename to exp0
    if os.path.exists(os.path.join(all_results_folder, 'exp')):
        os.rename(os.path.join(all_results_folder, 'exp'), os.path.join(all_results_folder, 'exp0'))
        
    # Get the latest experiment number using from the runs/detect folder
    exp = max([int(re.search('\d+', f.name).group()) for f in os.scandir(all_results_folder) if f.is_dir()])

    # Get the path to the latest experiment folder
    results_folder = os.path.join(all_results_folder, f'exp{exp}', 'labels')

    # iterate over all text files in labels folder
    for file in os.listdir(results_folder):
        # extract bounding boxes and class labels from text file
        with open(os.path.join(results_folder, file), 'r') as f:
            data = f.readlines()
            data = [x.strip() for x in data]
            data = [x.split(' ') for x in data]
        
        print(data)
