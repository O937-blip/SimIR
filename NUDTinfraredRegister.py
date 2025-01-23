import os
import base64

image_directory = ''
annotation_directory = ''
tsv_file = ''
index_file = ''

# Get a list of all files in the image directory
image_file_list = os.listdir(image_directory)

with open(tsv_file, 'w') as f1, open(index_file, 'w') as f2:
    ann_start = 0

    # Iterate through the list of image files
    for image_file_name in image_file_list:
        # Check if the file is an image file (e.g., JPG)
        if image_file_name.endswith('.png'):
            image_file = os.path.join(image_directory, image_file_name)

            # Construct the corresponding annotation file path
            annotation_file_name = image_file_name.replace('.png', '.png')  # Adjust the extension if needed
            annotation_file = os.path.join(annotation_directory, annotation_file_name)

            # Check if the matching annotation file exists
            if os.path.isfile(annotation_file):
                # Read image data and encode in base64
                img = open(image_file, 'rb').read()
                img = base64.b64encode(img).decode('utf-8')

                # Read annotation data and encode in base64 (adjust the reading method accordingly)
                ann = open(annotation_file, 'rb').read()
                ann = base64.b64encode(ann).decode('utf-8')

                # Save image file name
                length = f1.write("%s\t" % image_file)

                # Save annotation data
                length += f1.write("%s\t" % ann)

                # Save image
                length += f1.write("%s\n" % img)

                # Save the index information
                f2.write("%d %d\n" % (ann_start, length))
                ann_start += length
