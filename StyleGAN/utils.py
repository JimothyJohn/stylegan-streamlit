#!/usr/bin/python3
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example program shows how to find frontal human faces in an image and
#   estimate their pose.  The pose takes the form of 68 landmarks.  These are
#   points on the face such as the corners of the mouth, along the eyebrows, on
#   the eyes, and so forth.
#
#   The face detector we use is made using the classic Histogram of Oriented
#   Gradients (HOG) feature combined with a linear classifier, an image pyramid,
#   and sliding window detection scheme.  The pose estimator was created by
#   using dlib's implementation of the paper:
#      One Millisecond Face Alignment with an Ensemble of Regression Trees by
#      Vahid Kazemi and Josephine Sullivan, CVPR 2014
#   and was trained on the iBUG 300-W face landmark dataset (see
#   https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/):  
#      C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic. 
#      300 faces In-the-wild challenge: Database and results. 
#      Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.
#   You can get the trained model file from:
#   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2.
#   Note that the license for the iBUG 300-W dataset excludes commercial use.
#   So you should contact Imperial College London to find out if it's OK for
#   you to use this model file in a commercial product.

import os, argparse, glob, json, dlib
from pathlib import Path
from ffhqtools import recreate_aligned_images

def cleanJSON(json_template):
    jsonClone = json_template
    jsonClone["0"]["metadata"]["photo_url"]=""
    jsonClone["0"]["metadata"]["photo_title"]=""
    jsonClone["0"]["metadata"]["author"]="Nick Armenta"
    jsonClone["0"]["metadata"]["country"]="USA"
    jsonClone["0"]["metadata"]["date_uploaded"]=""
    jsonClone["0"]["metadata"]["date_crawled"]="N/A"
    
    jsonClone["0"]["image"]["file_url"]=""
    jsonClone["0"]["image"]["file_path"]=""
    jsonClone["0"]["image"]["file_size"]=0
    jsonClone["0"]["image"]["file_md5"]=""
    jsonClone["0"]["image"]["pixel_md5"]=""
    jsonClone["0"]["image"]["face_landmarks"]=[[0, 0] for count in range(68)]

    jsonClone["0"]["thumbnail"]["file_url"]=""
    jsonClone["0"]["thumbnail"]["file_path"]=""
    jsonClone["0"]["thumbnail"]["file_size"]=0
    jsonClone["0"]["thumbnail"]["file_md5"]=""
    jsonClone["0"]["thumbnail"]["pixel_md5"]=""

    jsonClone["0"]["in_the_wild"]["file_url"]=""
    jsonClone["0"]["in_the_wild"]["file_path"]=""
    jsonClone["0"]["in_the_wild"]["file_size"]=0
    jsonClone["0"]["in_the_wild"]["file_md5"]=""
    jsonClone["0"]["in_the_wild"]["pixel_md5"]=""
    jsonClone["0"]["in_the_wild"]["face_rect"]=[0, 0, 0, 0]
    jsonClone["0"]["in_the_wild"]["face_landmarks"]=[[0, 0] for count in range(68)]
    jsonClone["0"]["in_the_wild"]["face_quad"]=""

    return jsonClone

def get_landmarks(dlibImage):
    # Find bounding box of faces and upsample once
    dets = detector(dlibImage, 1)
    if len(dets)==0: return None # if no faces are found report none
    detection = dets[0] # CHANGE THIS LINE TO LOOP FOR MULTIPLE FACES
    shape = predictor(dlibImage, detection) # get face landmarks
    # Create landmark list
    return [(part.x, part.y) for part in shape.parts()]

SAVING_JSON = True

def AlignFace(input, jsonFilename):
    # Create output directory if needed
    if not os.path.exists('out/json'): os.mkdir('out/json')

    # Load json template and duplicate
    with open('templates/template.json', 'r') as f:
        f_json = json.load(f)
        f.close()

    newJson = cleanJSON(f_json)

    # Initialize dlib objects
    detector = dlib.get_frontal_face_detector()
    faces = dlib.full_object_detections()
    predictor = dlib.shape_predictor(args.predictor_path)

    # For every image in the target directory
    if len(glob.glob(os.path.join(args.faces_folder_path, "*.*")))==0:
        print('No input files available!')
        exit()

    for f in sorted(glob.glob(os.path.join(input, "*.*"))):
        num = str(0)
        fileName = Path(f).stem # extract filename

        # Save metadata
        img = dlib.load_rgb_image(f) # load image for dlib

        # Find bounding box of faces and upsample once
        dets = detector(img, 1)
        if len(dets)==0:
            print(f"Skipping image {f}")
            continue # if no faces are found continue
        
        print(f"Aligning image {f}")
        detection = dets[0] # CHANGE THIS LINE TO LOOP FOR MULTIPLE FACES
        shape = predictor(img, detection) # get face landmarks
        # Create landmark list
        landmarks = [(part.x, part.y) for part in shape.parts()]
        newJson[num]["in_the_wild"]["face_landmarks"] = landmarks # save landmarks to json
        newJson[num]["in_the_wild"]["face_rect"]=[detection.left(),
                                                detection.top(),
                                                detection.right(),
                                                detection.bottom()]
        newJson[num]["in_the_wild"]["file_path"] = str(f)
        newJson[num]["in_the_wild"]["file_size"]= os.path.getsize(f)

        # Align to landmarks in json file
        alignedImage = recreate_aligned_images(newJson,
                                            output_size=1024,
                                            transform_size=4096,
                                            enable_padding=True)

        # Save aligned image
        savePath = "{}/{}.png".format('/out', fileName)
        alignedImage.save(savePath) # save image to file
        if SAVING_JSON:
            newJson[num]["image"]["file_path"] = savePath
            newJson[num]["image"]["file_size"]= os.path.getsize(savePath)
            # CHANGE TO USE IMAGE VARIABLE
            newJson[num]["image"]["face_landmarks"] = get_landmarks(dlib.load_rgb_image(savePath))

            with open("out/json/{}.json".format(jsonFilename), 'w') as f:
                json.dump(newJson, f)
                f.close()
