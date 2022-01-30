# Checklist

## To Be Inspected:
- [x] dataset:
    - [x] input image size and number of channels: (720, 1280, 3), constant for every image
    - [x] labels: [{...}, {...}, {...}, etc.] for each image, with {...} in variable quantity ∈ [0 ; +∞), where {...} ≡ {'x': ..., 'y': ..., 'width': ..., 'height': ...} ⇒ a single class is considered (coral-eating crown-of-thorns starfish)
    - [x] temporal information: available, in terms of previous/following frame
    - [x] number of training-plus-validation images: 23501
    - [x] number of test images: ~ 13000

## To Be Implemented:
- [ ] data augmentation
- [ ] cross validation
- [ ] checkpoint averaging
- [ ] post processing to turn model outputs into predictions of boundinx boxes
- [ ] run many experiments


# Observations for Optimization

## Unnecessary Characteristics:
- [x] model:
    - [x] no need to support more classes
    - [x] no need to support variable input sizes

## Necessary Characteristics:
- [x] inputs:
    - [x] need to keep a high enough resolution, since coral-eating crown-of-thorns starfishes look difficult to distiguish from the surrounding background even to the human eye - assumed to be the gold standard creating labels in this problem - and a lower resolution would confuse even the gold standard (after manual inspection)
    - [x] need for many anchors, as coral-eating crown-of-thorns starfishes are presented as very small in the training/validation samples (after manual inspection)
