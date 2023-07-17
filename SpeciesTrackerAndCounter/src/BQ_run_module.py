import os
# from SpeciesDetector.src.detect import *
from track import *
from globox.src import globox
from pathlib import Path
from collections import defaultdict
import argparse
def convert_label(input_path, output_path):
    label_path = Path(input_path)  # Where the .txt files are
    image_path = Path(input_path)  
    save_file = Path(output_path)

    # names_file = Path("/home/bowen68/projects/prairie_dog/data/yolo.names")
    # id_to_label = globox.AnnotationSet.parse_names_file(names_file)
    id_to_label = {
    '0': 'fragile pink urchin',
    '1': 'gray gorgonian',
    '2': 'squat lobster',
    '3': 'Basket star',
    '4': 'Long legged sunflower star',
    '5': 'Yellow gorgonian',
    '6': 'White slipper sea cucumber',
    '7': 'White spine sea cucumber',
    '8': 'Red swiftia gorgonian',
    '9': 'UI laced sponge',
    }

    label_to_id = {v: int(k)+1 for k, v in id_to_label.items()}
    # print(label_to_id)
    # imageid_to_id = {im: i for i, im in enumerate(sorted(self.image_ids))}
    
    # train
    annotations = globox.AnnotationSet.from_yolo_v5(folder=label_path, image_folder=image_path).map_labels(id_to_label)
    imageid_to_id = {im: i for i, im in enumerate(sorted(annotations.image_ids))}

    annotations.show_stats()
    annotations.save_coco(save_file, label_to_id =label_to_id, imageid_to_id=imageid_to_id, auto_ids=True, verbose=True)

def convert_to_cvat_video(input_path, output_path):
    id_to_label = {
    0: 'fragile pink urchin',
    1: 'gray gorgonian',
    2: 'squat lobster',
    3: 'basket star',
    4: 'long legged sunflower star',
    5: 'yellow gorgonian',
    6: 'white slipper sea cucumber',
    7: 'white spine sea cucumber',
    8: 'red swiftia gorgonian',
    9: 'UI laced sponge',
    }

    # input_path = '/home/bowen68/projects/bisque/Modules/SpeciesTrackerAndCounter/src/examples/output/example_more.txt'
    # frame_num, class_id, tracker_id, *box_xyxy (x_min y_min x_max y_max), confidence
    width = 1920
    height = 1080
    track_dict = defaultdict(list)
    with open(input_path, 'r') as f:
        yolo_annotations = f.readlines()

    for line in yolo_annotations:

        staff = line.split()  
        frame_id = int(staff[0])
        class_id = int(staff[1])
        tracker_id = int(staff[2])
        xtl, ytl, xbr, ybr = int(staff[3]), int(staff[4]), int(staff[5]), int(staff[6])


        track_dict[tracker_id].append([frame_id, class_id, xtl, ytl, xbr, ybr])

    output_file = open(output_path, 'w')
    output_file.write('<annotations>\n')
    for track_id, item in track_dict.items():
        if len(item) != 1:
            label_name = id_to_label[item[0][1]]
            string = f'<track id=\"{track_id}\" label=\"{label_name}\">\n'
            output_file.write(string)
        
            for i, (frame_id, class_id, xtl, ytl, xbr, ybr) in enumerate(item):
                outside = 0
                string = f'<box frame=\"{frame_id-1}\" keyframe=\"1\" outside=\"{outside}\" occluded=\"0\" xtl=\"{xtl}\" ytl=\"{ytl}\" xbr=\"{xbr}\" ybr=\"{ybr}\" z_order=\"0\">\n</box>\n'

                if i != len(item)-1:
                    outside = 0
                else:
                    outside = 1
                string = f'<box frame=\"{frame_id-1}\" keyframe=\"1\" outside=\"{outside}\" occluded=\"0\" xtl=\"{xtl}\" ytl=\"{ytl}\" xbr=\"{xbr}\" ybr=\"{ybr}\" z_order=\"0\">\n</box>\n'
                output_file.write(string)
            
            output_file.write('</track>\n')
    output_file.write('</annotations>\n')
    output_file.close()


# input_path_dict will have input file paths with keys corresponding to the input names set in the cli.
def run_module(input_path_dict, output_folder_path, min_hysteresis=100, max_hysteresis=200):
    """
    This function should load input resources from input_path_dict, do any pre-processing steps, run the algorithm,
    save all outputs to output_folder_path, AND return the outputs_path_dict.
    
    :param input_path_dict: Dictionary of input resource paths indexed by input names. 
    :param output_folder_path: Directory where to save output results.
    :param min_hysteresis: Tunable parameter must have default values.
    :param max_hysteresis: Tunable parameter  must have default values.
    :return: Dictionary of output result paths.
    """
    
    ##### Preprocessing #####

    # Get input file paths from dictionary
    input_video_path = input_path_dict['Input Video'] # KEY MUST BE DESCRIPTIVE, UNIQUE, AND MATCH INPUT NAME SET IN CLI

    _, file_name = os.path.split(input_video_path)
    ##### Run algorithm #####
    video_output_path, hdf_path, anno_path = track(input_video_path)
    xml_path = anno_path.split('.')[0] + '.xml'
    convert_to_cvat_video(anno_path, xml_path)
    # ##### Save output #####
    
    # output_folder_path = '/module/src/runs/detect/detection/'
    # output_folder_path = '/home/bowen68/projects/bisque/Modules/SpeciesTrackCount/src/examples/output/'
    # output_file_path = os.path.join(output_folder_path, file_name)
    # print(output_file_path)
    # Create dictionary of output paths
    output_paths_dict = {}

    output_paths_dict['Output Video'] = video_output_path 
    output_paths_dict['Output Counts'] = hdf_path 
    output_paths_dict['YOLO Annotation File'] = anno_path
    output_paths_dict['CVAT Annotation File'] = xml_path
    # merge labels to one file
    # output_label_folder = '/module/src/runs/detect/detection/labels/'
    # read_files = glob.glob(output_label_folder + "*.txt")

    # with open("/module/src/runs/detect/detection/labels.txt", "wb") as outfile:
    #     for f in read_files:
    #         # print(f)
    #         file_name = os.path.basename(f)
    #         # print(file_name)
    #         with open(f, "rb") as infile:
    #             outfile.write(f'{file_name} \n'.encode())
    #             outfile.write(infile.read())
    
    # input_label_path = os.path.join(output_folder_path, 'labels')
    # output_label_path = os.path.join(output_folder_path, 'labels.json')
    # convert_label(input_label_path, output_label_path)

    # output_paths_dict['Annotation File'] = output_label_path
    ##### Return output paths dictionary #####  -> IMPORTANT STEP
    return output_paths_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    args = parser.parse_args()
    print(f'Running on {args.input}')

    # Place some code to test implementation
    
    # Define input_path_dict and output_folder_path
    input_path_dict = {}
    current_directory = os.getcwd()
    # Place test image in current directory
    input_path_dict['Input Video'] = os.path.join(current_directory, args.input) # KEY MUST MATCH INPUT NAME SET IN CLI
    # input_path_dict['Input Video'] = os.path.join(current_directory,'examples/example_more.mp4') # KEY MUST MATCH INPUT NAME SET IN CLI
    output_folder_path = current_directory
    
    # Run algorithm and return output_paths_dict
    output_paths_dict = run_module(input_path_dict, output_folder_path, min_hysteresis=100, max_hysteresis=200)
    
    # Get outPUT file path from dictionary
    output_video_path = output_paths_dict['Output Video'] # KEY MUST MATCH OUTPUT NAME SET IN CLI
    print(output_paths_dict)
    # Load data
    # out_image = cv2.imread(output_video_path, 0)
    # # Display output image and ensure correct output
    # cv2.imshow("Results",out_img)