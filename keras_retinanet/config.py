import keras
import numpy as np

DEFAULT_SCALES = [0.5, 1, 2]
DEFAULT_RATIOS = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

SCALES = [0.5, 0.7, 1, 1.5, 2]
RATIOS = [0.2, 0.33, 0.5, 0.62, 0.79, 1]

NUM_ANCHORS = len(SCALES) * len(RATIOS)


from .models.retinanet import AnchorParameters
ANCHOR_PARAMS = AnchorParameters(
    sizes=[32, 64, 128, 256, 512],
    strides=[8, 16, 32, 64, 128],
    ratios=np.array(RATIOS, keras.backend.floatx()),
    scales=np.array(SCALES, keras.backend.floatx()),
)

