# Checklist

# To Be Inspected:
- [] input image size and number of channels: (720, 1280, 3), constant for every image
- [] labels: [{...}, {...}, {...}, etc.] for each image, with {...} in variable quantity ∈ [0 ; +∞), where {...} ≡ {'x': ..., 'y': ..., 'width': ..., 'height': ...}
- [] temporal information: available, in terms of previous/following frame
- [] number of training-plus-validation images: 23501
- [] number of test images: ~ 13000

# To Be Implemented:
- [ ] cross validation
- [ ] checkpoint averaging
- [ ] run many experiments
