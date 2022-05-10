import numpy as np
import utils.color as color_module
import utils.draw as draw_module


def label_colormap(n_label=256, value=None):
    """Label colormap.

    Parameters
    ----------
    n_labels: int
        Number of labels (default: 256).
    value: float or int
        Value scale or value of label color in HSV space.

    Returns
    -------
    cmap: numpy.ndarray, (N, 3), numpy.uint8
        Label id to colormap.

    """

    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    cmap = np.zeros((n_label, 3), dtype=np.uint8)
    for i in range(0, n_label):
        id = i
        r, g, b = 0, 0, 0
        for j in range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b

    if value is not None:
        hsv = color_module.rgb2hsv(cmap.reshape(1, -1, 3))
        if isinstance(value, float):
            hsv[:, 1:, 2] = hsv[:, 1:, 2].astype(float) * value
        else:
            assert isinstance(value, int)
            hsv[:, 1:, 2] = value
        cmap = color_module.hsv2rgb(hsv).reshape(-1, 3)
    return cmap


def label2rgb(
        label,
        img=None,
        alpha=0.5,
        label_names=None,
        font_size=30,
        thresh_suppress=0,
        colormap=None,
        loc="centroid",
        font_path=None,
):
    """Convert label to rgb.

    Parameters
    ----------
    label: numpy.ndarray, (H, W), int
        Label image.
    img: numpy.ndarray, (H, W, 3), numpy.uint8
        RGB image.
    alpha: float
        Alpha of RGB (default: 0.5).
    label_names: list or dict of string
        Label id to label name.
    font_size: int
        Font size (default: 30).
    thresh_suppress: float
        Threshold of label ratio in the label image.
    colormap: numpy.ndarray, (M, 3), numpy.uint8
        Label id to color.
        By default, :func:`~imgviz.label_colormap` is used.
    loc: string
        Location of legend (default: 'centroid').
        'lt' and 'rb' are supported.
    font_path: str
        Font path.

    Returns
    -------
    res: numpy.ndarray, (H, W, 3), numpy.uint8
        Visualized image.

    """
    if colormap is None:
        colormap = label_colormap()

    res = colormap[label]

    random_state = np.random.RandomState(seed=1234)

    mask_unlabeled = label < 0
    res[mask_unlabeled] = random_state.rand(*(mask_unlabeled.sum(), 3)) * 255

    if img is not None:
        if img.ndim == 2:
            img = color_module.gray2rgb(img)
        res = (1 - alpha) * img.astype(float) + alpha * res.astype(float)
        res = np.clip(res.round(), 0, 255).astype(np.uint8)

    if label_names is None:
        return res

    unique_labels = np.unique(label)
    unique_labels = unique_labels[unique_labels != -1]
    if isinstance(label_names, dict):
        unique_labels = [l for l in unique_labels if label_names.get(l)]
    else:
        unique_labels = [l for l in unique_labels if label_names[l]]
    if len(unique_labels) == 0:
        return res

    if loc == "centroid":
        for label_i in unique_labels:
            mask = label == label_i
            if 1.0 * mask.sum() / mask.size < thresh_suppress:
                continue
            y, x = np.array(_center_of_mass(mask), dtype=int)

            if label[y, x] != label_i:
                Y, X = np.where(mask)
                point_index = np.random.randint(0, len(Y))
                y, x = Y[point_index], X[point_index]

            text = label_names[label_i]
            height, width = draw_module.text_size(
                text, size=font_size, font_path=font_path
            )
            color = color_module.get_fg_color(res[y, x])
            res = draw_module.text(
                res,
                yx=(y - height // 2, x - width // 2),
                text=text,
                color=color,
                size=font_size,
                font_path=font_path,
            )
    elif loc in ["rb", "lt"]:
        text_sizes = np.array(
            [
                draw_module.text_size(
                    label_names[l], font_size, font_path=font_path
                )
                for l in unique_labels
            ]
        )
        text_height, text_width = text_sizes.max(axis=0)
        legend_height = text_height * len(unique_labels) + 5
        legend_width = text_width + 20 + (text_height - 10)

        height, width = label.shape[:2]
        legend = np.zeros((height, width, 3), dtype=np.uint8)
        if loc == "rb":
            aabb2 = np.array([height - 5, width - 5], dtype=float)
            aabb1 = aabb2 - (legend_height, legend_width)
        elif loc == "lt":
            aabb1 = np.array([5, 5], dtype=float)
            aabb2 = aabb1 + (legend_height, legend_width)
        else:
            raise ValueError("unexpected loc: {}".format(loc))
        legend = draw_module.rectangle(
            legend, aabb1, aabb2, fill=(255, 255, 255)
        )

        alpha = 0.5
        y1, x1 = aabb1.round().astype(int)
        y2, x2 = aabb2.round().astype(int)
        res[y1:y2, x1:x2] = (
                alpha * res[y1:y2, x1:x2] + alpha * legend[y1:y2, x1:x2]
        )

        for i, l in enumerate(unique_labels):
            box_aabb1 = aabb1 + (i * text_height + 5, 5)
            box_aabb2 = box_aabb1 + (text_height - 10, text_height - 10)
            res = draw_module.rectangle(
                res, aabb1=box_aabb1, aabb2=box_aabb2, fill=colormap[l]
            )
            res = draw_module.text(
                res,
                yx=aabb1 + (i * text_height, 10 + (text_height - 10)),
                text=label_names[l],
                size=font_size,
                font_path=font_path,
            )
    else:
        raise ValueError("unsupported loc: {}".format(loc))

    return res


def _center_of_mass(mask):
    assert mask.ndim == 2 and mask.dtype == bool
    mask = 1.0 * mask / mask.sum()
    dx = np.sum(mask, 0)
    dy = np.sum(mask, 1)
    cx = np.sum(dx * np.arange(mask.shape[1]))
    cy = np.sum(dy * np.arange(mask.shape[0]))
    return cy, cx
