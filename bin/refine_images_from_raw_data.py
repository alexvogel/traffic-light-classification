import sys
import os
import argparse
import random
import yaml
import cv2

def create_cutouts(examples, output_dir, label_map):

    # toc file
    toc = {}

    # go through the data
    for example in examples:
        

        # read image
        img = cv2.imread(example['path'])

        #print("shape of whole image: " + str(img.shape) + "    " + example['path'])

        # for every annotation
        for i in range(0, len(example['annotations'])):

            annotation = example['annotations'][i]

            #print("xmin=" + str(int(annotation['xmin'])) + "  xmax=" + str(int(annotation['xmin']+annotation['x_width'])) + "   ymin=" + str(int(annotation['ymin'])) + "   ymax=" + str(int(annotation['ymin']+annotation['y_height'])))

            # create a cutout
            img_cutout = img[ int(annotation['ymin']) : int(annotation['ymin']+annotation['y_height']), int(annotation['xmin']) : int(annotation['xmin']+annotation['x_width']) ]

            # scale to 32x32 for use in classifier
            img_cutout_32_32 = cv2.resize(img_cutout, (32, 32))

            # get label name
            label_name = annotation['class']

            # set a path for the new image file
            outfile = output_dir + "/" + example['path'].split("/")[-1].split(".")[0] + "_" + str(i) + ".png"

            # det relative path for the entry in toc file
            outfile_rel_path = os.path.relpath(outfile, output_dir + "/..")

            # write toc
            toc[outfile_rel_path] = label_map[label_name]





            # log
            #print("shape of cutout image: " + str(img_cutout.shape))
           # print("shape of 32x32 cutout image: " + str(img_cutout_32_32.shape))
            
            # write cutout file
            cv2.imwrite(outfile ,img_cutout_32_32)

    return toc


def write_toc(filepath, toc):

    # open file for writing
    file = open(filepath, "w")

    # write header
    file.write("label,file\n")

    # write every key-value pair in a line of toc
    for path in toc.keys():
        file.write(str(toc[path]) + "," + path + "\n")


def write_label_map(filepath, label_map):

    # open file for writing
    file = open(filepath, "w")

    # write header
    file.write("class_id,class_name\n")

    # write every key-value pair in a line of toc
    for label in label_map.keys():
        file.write(str(label_map[label]) + "," + label + "\n")

def create_label_map(examples):

    # label map
    all_labels = {}

    # go through the data
    for example in examples:

        # for every annotation
        for i in range(0, len(example['annotations'])):

            annotation = example['annotations'][i]

            if annotation['class'] in all_labels.keys():
                all_labels[annotation['class']] += 1
            else:
                all_labels[annotation['class']] = 1

    label_list = all_labels.keys()

    # create a map name => id
    label_map = {}

    for i in range(0, len(label_list)):
        label_map[label_list[i]] = i+1

    return label_map



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_yaml", metavar='PATH', action='append', required=True, help="yaml file of udacity dataset.")
    parser.add_argument("--outdir", metavar='PATH', type=str, required=True, default="", help="output directory for traffic light images size 32x32")
    args = parser.parse_args()

    # det output directories
    outdir_images = args.outdir + "/images"

    # create directories
    if os.path.exists(args.outdir):
        print("error: delete or rename " + args.outdir)
        sys.exit(1)
    else:
        os.mkdir(args.outdir)
        os.mkdir(outdir_images)

    # join all yaml files concat
    yaml_strings = []

    for filepath in args.input_yaml:
        file = open(filepath, "rb")
        yaml_strings.append(file.read())

    #examples = yaml.load(open(args.input_yaml[0], 'rb').read())
    
    # for every yaml file
    all_examples = []

    # read in every yaml file in a seperate string
    for i in range(0, len(args.input_yaml)):

        examples_from_one_file = yaml.load(yaml_strings[i])

        print("Loaded " + str(len(examples_from_one_file)) + " examples from file " + args.input_yaml[i])

        # construct pathnames from yaml path and filename in yaml
        for j in range(len(examples_from_one_file)):
            examples_from_one_file[j]['path'] = os.path.abspath(os.path.join(os.path.dirname(args.input_yaml[i]), examples_from_one_file[j]['filename']))
    
        # extend the general examples list
        all_examples.extend(examples_from_one_file)

    # shuffle all_examples
    random.shuffle(all_examples)

    # print
#    print("splitting examples in " + str(len(examples_train)) + " for training and " + str(len(examples_test)) + " for testing")

    # create a label map ( name => id)
    label_map = create_label_map(all_examples)

    # write label map file
    write_label_map(args.outdir + "/label_map.csv", label_map)

    # create cutouts for train
    toc = create_cutouts(all_examples, outdir_images, label_map)

    # write toc train to file
    write_toc(args.outdir + "/toc.csv", toc)





if __name__ == '__main__':
	main()
