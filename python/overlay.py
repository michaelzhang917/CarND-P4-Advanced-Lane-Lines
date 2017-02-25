from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import cv2
import numpy as np
import matplotlib.pyplot as plt

def overlay_information(img, lanes):
    height, width, _ = img.shape

    left_curvature = lanes.left.meters.curvature(height)
    right_curvature = lanes.right.meters.curvature(height)
    distance_from_center = lanes.distance_from_center(center=(width / 2, height))

    img = overlay_text(img, "Left curvature: {0:.2f}m".format(left_curvature), pos=(10, 10))
    img = overlay_text(img, "Right curvature: {0:.2f}m".format(right_curvature), pos=(10, 90))
    img = overlay_text(img, "Distance from center: {0:.2f}m".format(distance_from_center), pos=(10, 180))
    return img

def overlay_lane(image, left_fit, right_fit, M):
    left_ys = np.linspace(0, 100, num=101) * 7.2
    left_xs = left_fit[0]*left_ys**2 + left_fit[1]*left_ys + left_fit[2]

    right_ys = np.linspace(0, 100, num=101) * 7.2
    right_xs = right_fit[0]*right_ys**2 + right_fit[1]*right_ys + right_fit[2]

    color_warp = np.zeros_like(image).astype(np.uint8)

    pts_left = np.array([np.transpose(np.vstack([left_xs, left_ys]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_xs, right_ys])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    newwarp = cv2.warpPerspective(color_warp, M, (image.shape[1], image.shape[0]))

    return cv2.addWeighted(image, 1, newwarp, 0.3, 0)


def overlay_text(image, text, pos=(0, 0), color=(255, 255, 255)):
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("../fonts/librefranklin-light.ttf", 64)
    draw.text(pos, text, color, font=font)
    image = np.asarray(image)

    return image

def overlay_detected_lane_data(image, lanes, M, showresults=False):
    height, width, _ = image.shape

    image = overlay_lane(image, lanes.left.pixels.fit, lanes.right.pixels.fit, M)
    image = overlay_text(image, "Left curvature: {0:.2f}m".format(lanes.left.meters.curvature(height)), pos=(10, 10))
    image = overlay_text(image, "Right curvature: {0:.2f}m".format(lanes.right.meters.curvature(height)), pos=(10, 90))
    image = overlay_text(image, "Distance from center: {0:.2f}m".format(lanes.distance_from_center((width/2, height))), pos=(10, 170))
    if showresults:
        f = plt.figure()
        plt.imshow(image)
        plt.show()
        f.savefig('../output/overlay.jpg', dpi=f.dpi, bbox_inches='tight')
    return image