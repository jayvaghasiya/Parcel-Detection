# Autodistill :
 
"Auto matic labeling of datset for instance segmentation and object detection tasks"

Steps :

1. Put all image that you want to label in images folder (in the same directory of sript)

2. Execute these commands:
 
    # For Instance-Sagmentation dataset
        python autodistill-labeling.py segmentation

    # For object-detection dataset
        python autodistill-labeling.py  
        OR
        python autodistill-labeling.py detection
        

# Note : You might need to change the onltology (The object you want to detect and it's label) as per your need.