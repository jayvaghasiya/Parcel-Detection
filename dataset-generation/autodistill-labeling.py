import os
import sys
import supervision as sv  # supervision for handling images and videos
from autodistill.detection import CaptionOntology
if len(sys.argv) == 2:
    argument = sys.argv[1]
    if argument == "segmentation":
        ## For instance sagementation dataset
        from autodistill_grounded_sam import GroundedSAM as model
        print("Loading SAM Model For Instance-Sagmentation !!!")
    elif argument == "detection":
        ## For detection dataset
        from autodistill_grounding_dino import GroundingDINO as model
        print("Loading DINO Model For Object-Detection!!!")

    else:
        print("Invalid argument. Please Enter either 'segmentation' or 'detection'.")
else:
    ## For detection dataset
    from autodistill_grounding_dino import GroundingDINO as model
    print("Loading DINO Model For Object-Detection!!!")


#Getting the path of working directory
HOME = os.getcwd()
print(HOME)
IMAGE_DIR_PATH = f"{HOME}/images"

##for printing total number of images in our dataset
image_paths = sv.list_files_with_extensions(
    directory=IMAGE_DIR_PATH,
    extensions=["png", "jpg", "jpeg"])

print('image count :', len(image_paths))

IMAGE_DIR_PATH = f"{HOME}/images" 


## autodistill will label our object(cardboard box) as box
ontology=CaptionOntology({
    "tyre": "tyre",
})

DATASET_DIR_PATH = f"{HOME}/dataset" 

##Train the model to label our dataset
base_model = model(ontology=ontology)
dataset = base_model.label(
    input_folder=IMAGE_DIR_PATH,
    extension=".jpg",
    output_folder=DATASET_DIR_PATH)

print("-----------------------> Datset Labeled successfully !!!")
