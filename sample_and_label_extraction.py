"""
TODO
"""


from csv import reader as csv_reader
from json import loads as json_loads
from os import getcwd
from os.path import join as path_join
from typing import Dict, List

# pylint: disable=import-error
from tensorflow.data import AUTOTUNE, Dataset
# pylint: enable=import-error


DATASET_DIR = path_join(
    getcwd(),
    'tensorflow-great-barrier-reef'
)
LABELS_FILE_PATH = path_join(
    DATASET_DIR,
    'train.csv'
)


def label_line_into_image_path_to_bounding_boxes_dict(
        csv_label_line_segments: List[str]
) -> Dict[str, List[Dict[str, int]]]:
    """
    Turn any line of the CSV labels file from the original format
    'video_id,sequence,video_frame,sequence_frame,image_id,annotations' into a
    dictionary with the respective image file path as key and the respective
    bounding boxes as value.
    """
    image_path = path_join(
        DATASET_DIR,
        'train_images',
        'video_' + csv_label_line_segments[0],
        csv_label_line_segments[2] + '.jpg'
    )
    bounding_boxes = json_loads(
        csv_label_line_segments[5]
        .replace('"', '"""')
        .replace("'", '"')
    )

    return {
        image_path: bounding_boxes
    }


def load_labels_as_paths_to_bounding_boxes_dict() -> Dict[
        str, List[Dict[str, int]]
]:
    """
    Load the labels' information from the CSV file and return them as a
    dictionary associating image file paths to respective boundinx boxes.
    """
    image_paths_to_bounding_boxes = {}

    with open(LABELS_FILE_PATH, 'r') as labels_file:
        labels_reader = csv_reader(
            labels_file,
            delimiter=',',
            quotechar='"'
        )

        for line_index, line_segments in enumerate(labels_reader):
            if line_index == 0:
                continue

            # turning the label from the raw format into a processed
            # dictionary to retrieve bounding boxes of images easily from
            # respective image file paths:
            image_paths_to_bounding_boxes.update(
                label_line_into_image_path_to_bounding_boxes_dict(
                    csv_label_line_segments=line_segments
                )
            )

    return image_paths_to_bounding_boxes


IMAGE_PATHS_TO_BOUNDING_BOXES = load_labels_as_paths_to_bounding_boxes_dict()


# .cache().prefetch(buffer_size=AUTOTUNE)
# lambda image_path: paths_to_bounding_boxes[image_path]


if __name__ == '__main__':
    image_paths_dataset = Dataset.from_tensor_slices(
        tensors=[*IMAGE_PATHS_TO_BOUNDING_BOXES]  # only keys included
    )

    for i, path in enumerate(image_paths_dataset):
        if i < 5: print(path)
