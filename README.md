
# Image Search Engine

A Simple Search Engine for Irina Holidays Dataset by using MobileNetV2 to extract feature vectors. 


## Dependencies

* Python 3.10.11 is needed before installing program.
* Any Python IDE can be use for this project, recommend: VSCode and Pycharm
* OS: Windows 10 and 11
## Installation

Install my project with these following libraries in virtual environment (venv):

```bash
  pip install pandas
  pip install opencv-python
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  pip install numpy
  pip install tqdm
```
    
## Dataset

#### Dataset Link

```
  https://www.kaggle.com/datasets/vadimshabashov/inria-holidays
```

#### About Dataset
The Holidays dataset is a set of holidays photos. The dataset includes a very large variety of scene types (natural, man-made, water and fire effects, etc) and images are in high resolution.

The dataset contains 500 image groups, each of which represents a distinct scene or object.

Structure:

* images contains images
* groundtruth.json contains labels for each scene in the form:
```
  'scene_number':
        'query': image_path,
        'similar': [path to similar images]
```
File ` groundtruth.json ` was obtained by processing the names of the files (by authors: "the first image of each group is the query image and the correct retrieval results are the other images of the group").

#### Extract all the images into folders
We will extract all the image into seperate folders based on ` groundtruth.json `

* Query folder

| Folder name | Description                                   |
| :---------- | :------------------------------------------   |
| `input` | Contains all images with the tag `query` in json file |

* Database folder

| Folder name | Description                                   |
| :---------- | :-------------------------------------------  |
| `images` | Contains all images with the tag `similar` in json file|

* Result folder

| Folder name | Description                                   |
| :---------- | :------------------------------------------   |
| `output` | An empty folder, use to store the result images after retrieval|

After creating `input` and `output` folders, run `split.py` to move all query images to input folders.
```bash
  python split.py
```


## Deployment

After setting up all folders and installing all necessary libraries, use bellow command to run the program

```bash
  python -m main --input <path to an image in input folder>
```

Example: 
```bash
python -m main --input .\input\112400.jpg
```
After execution, all retrieved images will be stored in `output` folder and runtime, Recall, Precision, F1 score and other statistics are recorded in `result.txt`.

Furthermore, all images's name and semantic features are stored in `names.pkl` and `feature.pkl` respectively.

#### NOTE: 
if you run program without having any mentioned `.pkl` files, `names.pkl` and `feature.pkl` will be generated after execution. However, the program is going to retrieve all similar images base on these `.pkl` files if you already have it in project folder.

## Result Screenshots
All statistics after execution this command ` python -m main --input .\input\112400.jpg `


![Screenshot 2023-11-13 153305](https://github.com/Hung369/HCMUS-Simple_Image_Search_Engine/assets/81510821/3ac75f68-5a0b-491e-8216-2ad04d6f79ff)

## Demo
Video demo link: https://www.loom.com/share/ca31102067e24684aa93d4ef317de8a1?sid=970cba5c-1f8a-4950-8978-f35a221f2bdb

## Authors

[@manhhung](https://github.com/Hung369)

