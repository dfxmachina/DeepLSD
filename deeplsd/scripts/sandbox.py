import cv2
from fire import Fire
import numpy as np
from deeplsd.predictor import Predictor

def angular_diff(a, b):
    ac = np.exp(1j * np.array(a))
    bc = np.exp(1j * np.array(b))

    clockwise = np.angle(bc / ac) % np.pi
    counterclockwise = np.pi - clockwise

    diffs = np.minimum(clockwise, counterclockwise)
    return diffs


def main(path, ):
    image = cv2.imread(path)

    pred = Predictor()
    line_neighborhood = 5
    pred.set_custom_params({'grad_thresh': 3.0, })
    # the higher grad_thresh, the more angles are effectively zeroes out
    pred.set_custom_params({'line_neighborhood': line_neighborhood}, key=None)
    lines, out = pred.predict(image, with_other=True)
    img = pred.draw_lines(image, lines.round().astype(int))

    # angle, = out['line_level']
    # angle_vis = np.zeros_like(image)
    #
    # cos = (np.cos(angle) + 1) * 127.5
    # sin = (np.sin(angle) + 1) * 127.5
    #
    # angle_vis[:, :, 0] = cos
    # angle_vis[:, :, 1] = sin
    # angle_vis = angle_vis.astype("uint8")

    image_other = cv2.imread(path)

    df = out['df'].squeeze(0).cpu().numpy()
    ll = out['line_level'].squeeze(0).cpu().numpy()

    angle_diff = angular_diff(ll, np.pi / 2)
    mask = np.abs(angle_diff) < np.pi / 12
    df[mask] = line_neighborhood

    def extract_lines(_df, _ll):
        return pred.net.detect_afm_lines(cv2.cvtColor(image_other, cv2.COLOR_BGR2GRAY),
                                         _df, _ll, **pred.net.conf['line_detection_params'])

    new_lines = extract_lines(df, ll)
    img_other = pred.draw_lines(image_other, new_lines.round().astype(int))

    img = np.hstack((img, img_other, ))

    # Display the results
    cv2.imshow("Original Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    Fire(main)
