import logging
import numpy as np
from enum import Enum, unique

logger = logging.getLogger()
logger.setLevel(logging.INFO)


@unique
class MarkTypes(Enum):
    ORIGINAL = "original"
    TRIANGLE = "triangle"
    SIN_HORIZON = "sin_horizon"
    SIN_VERTICAL = "sin_vertical"
    ELLIPSIS = "ellipsis"


RESOLUTION = 0.2


def allocate_supervised_data(alpha, n_total_clients, tr_label, ts_label):
    """
    allocate data to each client through Dirichlet distribution
    :param alpha: parameter of the Dirichlet distribution. the larger, the more uniform the # of samples per label
    on each client.
    :param n_total_clients: # of clients
    :param tr_label: a tensor of training data. [n_sample]
    :param ts_label: a tensor of test data. [n_sample]
    :return: client_idx_tr_samples: a list of tr samples index: [[[sample index of label 0],[],...,[]]
                                                               , [clietn 2]
                                                               , ...
                                                               , []]
            , client_idx_ts_samples: a list of tr samples index
    """
    # count samples for each class
    n_class = tr_label.unique().shape[0]
    logger.info(str(n_class) + " classes detected")

    n_tr_samples = np.zeros(n_class)
    n_ts_samples = np.zeros(n_class)
    for c in tr_label.unique().tolist():
        n_tr_samples[c] = tr_label[tr_label == c].shape[0]
        n_ts_samples[c] = ts_label[ts_label == c].shape[0]

    log_text = ["class " + str(c_id) + ": n_tr/n_ts: " + str(n_tr) + "/" + str(n_ts)
                for c_id, (n_tr, n_ts) in enumerate(zip(n_tr_samples, n_ts_samples))]
    logger.info("scale of samples per class:\n" + "\n".join(log_text))

    # generate sample distributions
    logger.info("generate class distributions over clients")
    prior = 1.0 * np.ones(n_total_clients) / n_total_clients
    class_dist = np.random.dirichlet(prior * alpha, n_class)  # row * col: n_class * n_client
    n_client_tr_samples = np.floor(class_dist.transpose() * n_tr_samples)  # row * col: n_client * n_class
    n_client_ts_samples = np.floor(class_dist.transpose() * n_ts_samples)  # row * col: n_client * n_class

    # allocate sample index to clients
    logger.info("allocate sample index to clients")
    idx_tr_samples = [np.random.permutation(range(int(n))) for n in n_tr_samples]  # permute tr sample index per class
    idx_ts_samples = [np.random.permutation(range(int(n))) for n in n_ts_samples]  # permute ts sample index per class

    client_idx_tr_samples = []
    client_idx_ts_samples = []
    for client_id, (tr_set, ts_set) in enumerate(zip(n_client_tr_samples, n_client_ts_samples)):
        cur_client_tr = []  # list store idx for current client  [[idx for class 0], ..., [], []]
        cur_client_ts = []  # list store idx for current client
        for cls_id, (n_sample_tr, n_sample_ts) in enumerate(zip(tr_set, ts_set)):
            cur_client_tr.append(idx_tr_samples[cls_id][:int(n_sample_tr)])  # allocate tr samples
            idx_tr_samples[cls_id] = idx_tr_samples[cls_id][int(n_sample_tr):]  # remove allocated samples from the list

            cur_client_ts.append(idx_ts_samples[cls_id][:int(n_sample_ts)])  # allocate ts samples
            idx_ts_samples[cls_id] = idx_ts_samples[cls_id][int(n_sample_ts):]  # remove allocated samples from the list

        client_idx_tr_samples.append(cur_client_tr)
        client_idx_ts_samples.append(cur_client_ts)

    return client_idx_tr_samples, client_idx_ts_samples


def generate_triangle_marks(idx, image_size, padding=1, degree=0.5, bias=1, resolution=RESOLUTION):
    degree = np.random.uniform()
    bias = np.random.randint(0, 1000)

    n_ch, n_row, n_col = image_size

    x_max, y_max = n_col - 1 - padding, n_row - 1 - padding
    x_min, y_min = 0 + padding, 0 + padding

    np.random.seed(bias)
    x_center = np.random.randint(x_min+x_max/4, x_max-x_max/4)
    np.random.seed(bias + 1)
    y_center = np.random.randint(y_min+y_max/4, y_max-y_max/4)
    r = min(x_center - x_min, x_max - x_center, y_center - y_min, y_max - y_center)

    while np.abs(degree - 0.) <= 0.1 or np.abs(degree - 0.33) <= 0.1 \
            or np.abs(degree - 0.66) <= 0.1 or np.abs(degree - 1.) <= 0.1:
        degree += 0.1
    degree_a = np.pi * degree
    degree_b = np.pi * (degree + 2./3.)
    degree_c = np.pi * (degree + 4./3.)

    a = (r * np.cos(degree_a)+x_center, r * np.sin(degree_a)+y_center)
    b = (r * np.cos(degree_b)+x_center, r * np.sin(degree_b)+y_center)
    c = (r * np.cos(degree_c)+x_center, r * np.sin(degree_c)+y_center)
    a, c, b = sorted([a, b, c], key=lambda x: x[0])

    rate_a_b = (b[1] - a[1]) / (b[0] - a[0])
    rate_a_c = (c[1] - a[1]) / (c[0] - a[0])
    rate_c_b = (b[1] - c[1]) / (b[0] - c[0])

    line_a_b = [(x, a[1]+(x-a[0])*rate_a_b) for x in np.arange(a[0], b[0], resolution)]
    line_a_c = [(x, a[1]+(x-a[0])*rate_a_c) for x in np.arange(a[0], c[0], resolution)]
    line_c_b = [(x, c[1]+(x-c[0])*rate_c_b) for x in np.arange(c[0], b[0], resolution)]

    cor = line_a_b + line_a_c + line_c_b
    cor = np.array(np.round(cor), int)

    return idx, cor


def generate_sin_marks(idx, image_size, padding=1
                       , A=None, phase=None, period=None
                       , vertical=False, resolution=RESOLUTION):
    """
    draw sin marks
    A=1., phrase=1/3, period=1., bias=14.
    :param image_size
    :param padding: the padding of x and y
    :param period: how many periods of the sin
    :param bias: bias of the Asin(wx+bias*2*pi)
    :param vertical:
    :return: image of the marks
    """
    if A is None:
        A = -1. * np.random.uniform(0.1, 0.9)
    if phase is None:
        phase = np.random.uniform()
    if period is None:
        period = np.random.uniform()

    n_ch, n_row, n_col = image_size
    x_max, y_max = n_col - 1 - padding, n_row - 1 - padding
    x_span = x_max - padding
    y_span = y_max - padding

    x_period = np.arange(0, 2 * np.pi*period, resolution) + 2 * np.pi * phase
    x_len = len(x_period)

    x = np.arange(padding, x_max, x_span/x_len)
    y = np.array([y_span / 2 + A * y_span / 2 * np.sin(z) for z in x_period]) + padding

    cor = np.zeros([len(x), 2], int)
    if vertical:
        cor[:, 0], cor[:, 1] = np.round(x), np.round(y)
    else:
        cor[:, 0], cor[:, 1] = np.round(y), np.round(x)

    return idx, cor


def generate_ellipse_marks(idx, image_size, padding=1, e=0.9, rot_angle=0.25, resolution=RESOLUTION):
    """
    draw ellipse marks
    :param
    :param padding: the padding of x and y
    :param e: eccentricity
    :param rot_angle: rotation angle
    :return: image with marks, image of the marks
    """
    rot_angle = np.random.uniform()
    n_ch, n_row, n_col = image_size
    x_max, y_max = n_col - 1 - padding, n_row - 1 - padding

    x_span = x_max - padding
    y_span = y_max - padding

    rot_angle = np.pi * rot_angle
    x_center = padding + int(x_span / 2)
    y_center = padding + int(y_span / 2)

    a = x_span / 2
    c = a * e
    b = np.sqrt(a ** 2 - c ** 2)

    # draw the ellipse
    t = np.arange(0, 2 * np.pi, resolution)
    x = np.cos(t) * a
    y = np.sin(t) * b

    # rotate
    xx = np.cos(rot_angle) * x - np.sin(rot_angle) * y
    yy = np.sin(rot_angle) * x + np.cos(rot_angle) * y

    # move the center
    xx += x_center
    yy += y_center

    cor = np.zeros([len(xx), 2], int)
    cor[:, 0], cor[:, 1] = np.round(yy), np.round(xx)

    return idx, cor


def generate_line_marks(idx, image_size, padding=1
                        , rate=None, bias_rate=None, resolution=RESOLUTION):
    """
    draw sin marks
    :param image_size
    :param padding: the padding of x and y
    :param rate:
    :return: image of the marks
    """
    if rate is None:
        rate = np.random.uniform(0.1, 0.4)
    rate_ver = 0.5 + rate

    if bias_rate is None:
        bias_rate = np.random.uniform(0.3, 0.7, 2)

    n_ch, n_row, n_col = image_size
    x_max, y_max = n_col - 1 - padding, n_row - 1 - padding
    x_min, y_min = padding, padding
    x_center = (x_max - x_min) * bias_rate[0] + x_min
    y_center = (y_max - y_min) * bias_rate[1] + y_min

    x_span = np.arange(x_min, x_max, resolution)
    y_span = np.array([np.tan(rate * np.pi) * (z - x_center) for z in x_span]) + y_center
    y_ver_span = np.array([np.tan(rate_ver * np.pi) * (z - x_center) for z in x_span]) + y_center
    # y_span, y_ver_span = y_span.clip(y_min, y_max), y_ver_span.clip(y_min, y_max)

    cor = np.zeros([len(x_span) * 2, 2], int)
    cor[:, 0], cor[:, 1] = np.round(np.concatenate([x_span, x_span])), np.round(np.concatenate([y_span, y_ver_span]))
    cor = cor[cor[:, 1] < y_max, :]
    cor = cor[cor[:, 1] > y_min, :]

    return idx, cor
