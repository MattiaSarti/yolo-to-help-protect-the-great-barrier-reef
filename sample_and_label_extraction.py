"""
TODO
"""


from csv import reader as csv_reader
from json import loads as json_loads
from os import getcwd
from os.path import join as path_join
from typing import Dict, List, Tuple
from unittest.mock import patch

from matplotlib.patches import Rectangle
from matplotlib.pyplot import pause as plt_pause, show as plt_show, subplots
# pylint: disable=import-error
from tensorflow import (
    convert_to_tensor, float32 as tf_float32, int64 as tf_int64, py_function,
    Tensor, uint8 as tf_uint8
)
from tensorflow.data import AUTOTUNE, Dataset
from tensorflow.io import decode_jpeg, read_file
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
        bytes(image_path, 'utf-8'): bounding_boxes
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


def load_sample_and_get_label(image_path: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Load the sample and get the label - representing bounding boxes - of the
    image represented by the input path.
    """
    return (
        decode_jpeg(
            contents=read_file(
                filename=image_path
            )
        ),
        convert_to_tensor(
            # bounding boxes as network output values:
            value=[
                [
                    bounding_box_dict['x'],
                    bounding_box_dict['y'],
                    bounding_box_dict['width'],
                    bounding_box_dict['height']
                ] for bounding_box_dict in
                IMAGE_PATHS_TO_BOUNDING_BOXES[image_path.numpy()]
            ],
            dtype=tf_int64
        )
    )


# TODO: .cache().prefetch(buffer_size=AUTOTUNE)
# TODO: .map() to preprocess sample vs preprocessing layer in the network?


if __name__ == '__main__':
    image_paths_dataset = Dataset.from_tensor_slices(
        tensors=[*IMAGE_PATHS_TO_BOUNDING_BOXES]  # only keys included
    )

    samples_and_labels_dataset = image_paths_dataset.map(
        map_func=lambda image_path: py_function(
            func=load_sample_and_get_label,
            inp=[image_path],
            Tout=(tf_uint8, tf_int64)
        ),
        num_parallel_calls=AUTOTUNE,  # TODO
        deterministic=True
    )

    # # l = [6, 43, 84, 6619]
    # # l = list(range(40, 50))
    # # d = {
    # #     l[0]: (0, 0),
    # #     l[1]: (0, 1),
    # #     l[2]: (1, 0),
    # #     l[3]: (1, 1),
    # # }
    # # _, axes = subplots(2, 2)
    # figure, axes = subplots(1, 1)
    # for i, sample_and_label in enumerate(samples_and_labels_dataset):
    #     if i in l:
    #         # print('.')
    #         # r, c = d[i]
    #         # l.remove(i)
    #         # axes[r, c].imshow(sample_and_label[0].numpy())
    #         axes.imshow(sample_and_label[0].numpy())
    #         for bounding_box in sample_and_label[1].numpy().tolist():
    #             # axes[r, c].add_patch(
    #             axes.add_patch(
    #                 p=Rectangle(
    #                     xy=(bounding_box[0], bounding_box[1]),
    #                     width=bounding_box[2],
    #                     height=bounding_box[3],
    #                     linewidth=2,
    #                     edgecolor='#00ff00',
    #                     facecolor='none'
    #                 )
    #             )
    #     show(block=False)
    #     pause(0.001)
    #     axes.clear()
    #     # if l == []:  # else:
    #     #     break

    figure, axes = subplots(1, 1)
    for sample_and_label in samples_and_labels_dataset:
        axes.clear()

        axes.imshow(sample_and_label[0].numpy())
        for bounding_box in sample_and_label[1].numpy().tolist():
            axes.add_patch(
                p=Rectangle(
                    xy=(bounding_box[0], bounding_box[1]),
                    width=bounding_box[2],
                    height=bounding_box[3],
                    linewidth=2,
                    edgecolor='#00ff00',
                    facecolor='none'
                )
            )

        plt_show(block=False)
        plt_pause(0.001)
