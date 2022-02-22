<h1 align="center">
    YOLO for a Kaggle Competition:<br>
    TensorFlow - Help Protect the Great Barrier Reef<br>
    🦈🐬🦭🐳🐋🐟🐠🐡🦑🐙🦐🦞🦀🐚🪨🌊🏝️🏖️🐢⛱️💦💧
</h1>

Unfortunately, I discovered and enrolled this wonderful competition too late, with only a minimal fraction of the whole competition time left.<br>
But I decided to take the opportunity of working with such a visually beautiful dataset anyway, to learn by **implementing** a new version of **YOLO** (specifically a variant of v3) myself **completely from scratch**, having fun exploring the dataset, defining the model, training it and evaluating it while knowing in advance there was not time for extensive hyperparameter tunings and for experimenting with different models.<br>
I froze all of my code as soon as the competition ended - this README is the only file that I kept modifying after my final submission - so this repository only contains what I quickly created over a bunch of days: a baseline of my model.<br>


## The Competition (Briefly)

[This](https://www.kaggle.com/c/tensorflow-great-barrier-reef) is the competition considered.<br>
The aim is to implement a single-class object detection algorithm, to distinguish a given kind of starfish in underwater video frames, that can be both trained and applyed for inferences on the test set within the computational limits given.<br>
The competition results were evaluated in terms of a metric that represents the F2 score, as they decided to favor recall twice as much as precision, avereaged over different IoU thresholds for considering bounding boxes as detected or not, with these thresholds being: {0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8}.<br>
I obtained a score of **TODO** on the competition test set, and a visual taste of what this score means in terms of the resulting model capabilities follows.<br>

#### A Video Example of My Model in Action:
<img style="width: 900px; height: auto; display: block; margin-left: auto; margin-right: auto;" class="animated-gif" src="https://github.com/MattiaSarti/yolo-to-help-protect-the-great-barrier-reef/raw/main/video-example.gif">


## This Repository Structure

Looking at the root directory of the repository:
    - the folder ***/kaggle*** contains [the command used to download the dataset](...), [the command used to create the submitted notebook assembling all the different source files](...) and [the final submitted notebook](...)
    - the folder ***/source*** contains all and only the source code compressed in the above-mentioned final submitted notebook, but split into different files for the sake of maintainability
    - the folder ***/docs/pictures*** is just a folder that contains small memories that give a taste of the competition, since the dataset can not be publicly shared
    - the file ***/docs/TODO.md*** is a collection of TODOs annotated and checked by myself during the competition solution development to note down and remember key thoughts of mine
    - the file ***/docs/README.md*** is what you are reading now


## Innovative Aspects

**TODO** anchors intended only as different relative shapes, in terms of ratio between width and height, to assign bounding boxes that could fall within the same cell to different positions in the anchor dimension of labels, letting the different model outputs for the different anchors specialize at predicting with different aspect ratios by themselves, as an alternative to the traditional strategy of forcing them to predict values on the same scale first and then rescaling them according to anchors


## ...

**NOTE**: I am sharing neither the competition dataset not any trained model among this repository's files, according to Kaggle's regulations. Only low-resolution screenshots of some example as displayed in this README. [link .gitignore]


## ...

```
Bounding Boxes' Statistics:
        - total number of bounding boxes: 11898
        - total number of images: 23501
        - average number of bounding boxes per image: 0.51
        - minimum number of bounding boxes per image: 0
        - maximum number of bounding boxes per image: 18
        - total number of empty images: 18582
        - average bounding box height [pixels]: 42.72
        - average bounding box width [pixels]: 47.89
        - average bounding boxes' centers distance [pixels]: 177.17
        - average bounding boxes' centers x-coord distance [pixels]: 130.74
        - average bounding boxes' centers y-coord distance [pixels]: 95.98
        - minimum bounding box height [pixels]: 13
        - minimum bounding box width [pixels]: 17
        - minimum bounding boxes' centers distance [pixels]: 3.04
        - minimum bounding boxes' centers x-coord distance [pixels]: 0.0
        - minimum bounding boxes' centers y-coord distance [pixels]: 0.0
        - maximum bounding box height [pixels]: 222
        - maximum bounding box width [pixels]: 243
        - maximum bounding boxes' centers distance [pixels]: 578.48
        - maximum bounding boxes' centers x-coord distance [pixels]: 565.5
        - maximum bounding boxes' centers y-coord distance [pixels]: 350.0
        - histogram of number of bounding boxes per image: see plot
        - histogram of bounding boxes' centers distance [pixels]: see plot
        - histogram of bounding boxes' centers x-coord distance [pixels]: see plot
        - histogram of bounding boxes' centers y-coord distance [pixels]: see plot
```
