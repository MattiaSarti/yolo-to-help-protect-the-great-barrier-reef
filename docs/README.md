<h1 align="center">
    YOLO for a Kaggle Competition:<br>
    TensorFlow - Help Protect the Great Barrier Reef<br>
    🦈🐬🦭🐳🐋🐟🐠🐡🦑🐙🦐🦞🦀🐚🪨🌊🏝️🏖️🐢⛱️💦💧
</h1>

Unfortunately, I discovered and enrolled this wonderful competition too late, with only a minimal fraction of the whole competition time left.<br>
But I decided to take the opportunity of working with such a visually beautiful dataset to learn by implementing a simplified version of YOLO (specifically a variant of v3) myself - completely from scratch - having fun exploring the dataset, defining the model, training and evaluating it while knowing in advance there was not time for extensive hyperparameter tunings and for experimenting with different models.<br>


## The Competition (Briefly)

[This](https://www.kaggle.com/c/tensorflow-great-barrier-reef) is the competition considered.<br>
The aim is to implement a single-class object detection algorithm, to distinguish a given kind of starfish in underwater video frames, that can be both trained and applyed for inferences on the test set within the computational limits given.<br>
The competition results were evaluated in terms of a metric that represents the F2 score, as they decided to favor recall twice as much as precision, avereaged over different IoU thresholds for considering bounding boxes as detected or not, with these thresholds being: {0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8}.<br>
I obtained a score of **TODO** on the competition test set, and a visual taste of what this score means in terms of the resulting model capabilities follows.<br>

#### A Video Example of My Model in Action:
**TODO**


## This Repository Structure

Looking at the root directory of the repository:
    - the folder ***/kaggle*** contains [the command used to download the dataset](...), [the command used to create the submitted notebook assembling all the different source files](...) and [the final submitted notebook](...)
    - the folder ***/source*** contains all and only the source code compressed in the above-mentioned final submitted notebook, but split into different files for the sake of maintainability
    - the folder ***/docs/pictures*** is just a folder that contains small memories that give a taste of the competition, since the dataset can not be publicly shared
    - the file ***/docs/TODO.md*** is a collection of TODOs annotated and checked by myself during the competition solution development to note down and remember key thoughts of mine
    - the file ***/docs/README.md*** is what you are reading now


## Innovative Aspects

**TODO** anchors intended only as different relative shapes, in terms of ratio between width and height, to assign bounding boxes that could fall within the same cell to different positions in the anchor dimension of labels, letting the different model outputs for the different anchors specialize at predicting with different aspect ratios by themselves, as an alternative to the traditional strategy of forcing them to predict values on the same scale first and then rescaling them according to anchors
