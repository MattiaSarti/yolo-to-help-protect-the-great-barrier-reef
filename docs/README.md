<h1 align="center">
    YOLO for a Kaggle Competition:<br>
    TensorFlow - Help Protect the Great Barrier Reef<br>
</h1>

Unfortunately, I discovered [this wonderful Kaggle competition](https://www.kaggle.com/c/tensorflow-great-barrier-reef) too late, when it was about to end. But I took the opportunity of working with such a visually pleasant dataset anyway to learn by **implementing and training** a new variant of **YOLO** (specifically of v3) myself **completely from scratch**.

<h3 align="center">
    🦈🐬🦭🐳🐋🐟🐠🐡🦑🐙🦐🦞🦀🐚🪨🌊🏝️🏖️🐢⛱️💦💧
</h3>

I enjoyed exploring the dataset, defining the model, training it and evaluating it, while knowing beforehand that there was not enough time for extensive hyperparameter tunings and for experimenting with different model architectures, preprocessing and postprocessing steps. I froze all of my code as soon as the competition ended - this README is the only file that I kept modifying after my final submission - so **this repository only contains what I quickly created over a bunch of days: a baseline model**.

#### A Video Example of My Model in Action:

<img align="left" style="width: auto; height: 440px; display: block; margin-left: auto; margin-right: auto;" class="animated-gif" src="https://github.com/MattiaSarti/yolo-to-help-protect-the-great-barrier-reef/raw/main/docs/pictures/video-example.gif">

<p align="right" style"width: auto; height: 440px;">
    <br><br><br><br>
    🟩 expected bounding boxes<br>(ground truth)
    <br><br><br>
    🟧 predicted bounding boxes,<br>opacity ∝ confidence
    <br><br><br><br>
</p>


## The Competition (Briefly)

[The competition](https://www.kaggle.com/c/tensorflow-great-barrier-reef) aim was to implement a single-class object detection algorithm to distinguish a given kind of starfish in underwater video frames - that could be both trained and applied for inferences on the test set within the given computational limits. Such test set predictions were evaluated in terms of a metric that represents the F2 score (as they decided to favor recall twice as much as precision) avereaged over different IoU thresholds for considering bounding boxes as detected correctly or not. I reached a test metric of 0.142, and a visual taste of what this score means in terms of the resulting model capabilities is displayed above.


## Innovative Aspects

Anchors - whom bounding boxes that could fall within the same cell are assigned to, for further differentiation - are here intended only as different relative shapes, in terms of ratio between bounding box width and height, letting the different model outputs for the different anchors specialize at predicting with different aspect ratios by themselves, as an alternative to the traditional strategy of forcing them to predict values on the same scale first and then rescaling them according to anchors.


## This Repository Structure

Looking at this repository:
- the folder [**/kaggle**](/kaggle) contains:
    - the [commands used to download the dataset](/kaggle/competition_data_download_commands.txt)
    - the [script employed to create the submitted notebook](/kaggle/generate_notebook_from_source.py) by assembling all the different source files
    - the [final, submitted notebook](/kaggle/submitted_notebook.ipynb)
- the folder [**/source**](/source) contains all and only the source code compressed in the above-mentioned final submitted notebook, but split into different files for the sake of maintainability, respectively for:
    - [common definitions](/source/common_constants.py)
    - [samples/labels extraction and inspection](/source/samples_and_labels.py)
    - [model architecture](/source/model_architecture.py)
    - [loss function and metric(s)](/source/loss_and_metrics.py)
    - [inference postprocessing](/source/inference.py)
    - [training and validation](/source/training_and_validation.py)
    - [running everything, from training to submitting predicitons on the test set](/source/main.py)
- the folder [**/docs**](/docs) contains:
    - a [collection of TODOs](/docs/TODO.md) annotated and checked by myself during the competition solution development to note down and remember key thoughts of mine
    - the [documentation](/docs/README.md) that you are reading now
    - a [subfolder](/docs/pictures) containing pictures and visual results saved for documentation purposes (and small memories that give a taste of the competition, since the dataset can not be publicly shared)


## Reproducing My Results

All results are perfectly reproducible since all the seeds of all the stochastic components at stake are fixed in a purely deterministic fashion. Python 3.8.0 was employed, and the required Python modules can be installed via pip 21.3.1 by running from the root folder of the repository the following command:
```
pip install -r requirements.txt
```
After installing such requirements,<br>
either:
- enter the [/kaggle](/kaggle) folder in the root of the repository and run the following command to generate the final notebook by assembling all the code spread across the different source files:
    ```
    generate_notebook_from_source.py
    ```
    and then load and execute the whole notebook as is on Kaggle (in the competition workspace)

or:
- enter the [/source](/source) folder in the root of the repository and run the following command to run everything locally, (assuming you have somehow access to the competition workspace to infer on the test set):
    ```
    python main.py
    ```


## Dataset & Bounding Boxes Statistics:

Inspected by running, from inside the repository folder **/source**, the following python commands...

<details>
<summary>Commands</summary>

```
from samples_and_labels import inspect_bounding_boxes_statistics_on_training_n_validation_set
inspect_bounding_boxes_statistics_on_training_n_validation_set()
```

</details>

...which print to standard outputand save to **/docs/pictures** respectively the following resume and the following visual plots, together summarizing all inspected statistics:

<details>
<summary>Statistics</summary>

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
<img src="/docs/pictures/Histogram of Bounding Boxes' Centers Distance [pixels].png">
<img src="/docs/pictures/Histogram of Bounding Boxes' Centers X-Coordinate Distance [pixels].png">
<img src="/docs/pictures/Histogram of Bounding Boxes' Centers Y-Coordinate Distance [pixels].png">
<img src="/docs/pictures/Histogram of Number of Bounding Boxes per Image.png">

</details>


## Compliance

I am sharing neither the competition dataset nor any trained model, according to Kaggle's rules (as confirmed [here](/.gitignore)). Only low-resolution pictures of few examples are employed for this documentation.
