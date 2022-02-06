# Checklist

## To Be Inspected:
- [x] dataset:
    - [x] input image size and number of channels: (720, 1280, 3), constant for every image
    - [x] labels: [{...}, {...}, {...}, etc.] for each image, with {...} in variable quantity ∈ [0 ; +∞), where {...} ≡ {'x': ..., 'y': ..., 'width': ..., 'height': ...} ⇒ a single class is considered (coral-eating crown-of-thorns starfish)
    - [x] temporal information: available, in terms of previous/following frame
    - [x] number of training-plus-validation images: 23501
    - [x] number of test images: ~ 13000
    - [x] bounding boxes statistics: see docs & code
- [x] preprocessing:
    - [x] what data augmentation? I have no domain knowledge ⇒ I can only assume that horizontal flips are a valid data augmentation schema, while vertical flips and any other image transformation (e.g. cropping, resizing, zooming, exposure/brightness/color changes, histogram stretchings, elastic deformations) could distort the task optimization
    - [x] which for of input image normalization is better? absolute (relative to the training set overall statistics) or relative (each sample normalized to have given mean and std)? learnable or fixed?

## To Be Implemented:
- [x] data loadage and splitting
- [x] data preprocessing
- [x] data augmentation
- [x] labels pre-processing (bounding boxes labels → model outputs labels)
- [x] model architecture
- [x] model training and validation
- [ ] loss function
- [ ] post-processing (model outputs → predictions of boundinx boxes)
- [ ] competition metric
- [ ] test set inference
- [x] fix determinism for reproducibility
- [ ] checkpoint averaging
- [ ] cross validation
- [ ] hyperparameter tuning
- [ ] many experiments


# Observations for Optimization

## Unnecessary Characteristics:
- [x] model:
    - [x] no need to support more classes ⇒ no need (for each anchor/grid part) to have different outputs for object presence score and class probability, they can be fused in a single output representing the probability of presence of coral-eating crown-of-thorns starfishes
    - [x] not reasonable to fine tune a model pretrained on another "general" dataset as the features necessary to detect coral-eating crown-of-thorns starfishes look - after manual inspection - very different from the patterns that allow to distinguish most "general" classes' objects
- [ ] training:
    - [ ] most frames do not contain bounding boxes, or only one ⇒ balance samples based on number of bounding boxes contained while training, but keep the original unbalance for validation, which reflects test-time conditions

## Necessary Characteristics:
- [ ] inputs:
    - [x] need to keep a high enough resolution, since coral-eating crown-of-thorns starfishes look difficult to distiguish from the surrounding background even to the human eye - assumed to be the gold standard creating labels in this problem - and a lower resolution would confuse even the gold standard (after manual inspection)
    - [x] color looks relevant to distinguish coral-eating crown-of-thorns starfishes (after manual inspection)
    - [ ] the temporal information is useful, despite not all frames have a previous one actually reflecting the previous timestamp of the same video sequence
- [x] model:
    - [x] no need to support variable input sizes (anchors) but they can be helpful anyway to have larger grid cells while often having very close objects in the dataset (after manual inspection)
    - [x] need for a model capable on outputting close bounding boxes, as coral-eating crown-of-thorns starfishes are presented as very small and close in the training/validation samples (after manual inspection)
    - [x] since - after manual inspection - often coral-eating crown-of-thorns starfishes appear very small and very close within the same video frame, the model architecture must allow for an acute enough resolution
