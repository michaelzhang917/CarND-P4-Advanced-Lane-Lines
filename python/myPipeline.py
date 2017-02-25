import matplotlib.image as mpimg
import pickle
from lane import *
from  overlay import  *
from moviepy.editor import VideoFileClip



# performs the camera calibration, image distortion correction and
# returns the undistorted image
def cal_undistort(img, mtx, dist, showResults=False, filename=[]):
    # Use cv2.calibrateCamera and cv2.undistort()
    img_size = (img.shape[1], img.shape[0])
    # Find the chessboard corners
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    if showResults:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(undist)
        ax2.set_title('Undistorted Image', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()
        if len(filename) != 0 :
            f.savefig(filename, dpi=f.dpi)
    return undist

# A function that applies Sobel x or y,
# then takes an absolute value and applies a threshold.

def abs_sobel_thresh(img, sobel_kernel=3, orient='x', thresh=(0,255)):

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output


# A function that calculate the total magnitude of Sobel x or y,
# then applies a threshold.

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return binary_output

# A function that calculate the angle of Sobel x or y,
# then takes an absolute value and applies a threshold.

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output


def SChannel_threshold(img, thresh=(0,255)):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hsv[:, :, 2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

def RChannel_threshold(img, thresh=(0,255)):
    r_channel = img[:, :, 0]
    binary_output = np.zeros_like(r_channel)
    binary_output[(r_channel >= thresh[0]) & (r_channel <= thresh[1])] = 1
    return binary_output

def combined_threshold(img, showresults=False, filename=[]):
    """Find and return a binary image based on the combination of thresholds.
    Uses an optional region of interest polygon.
    """
    comb_img = np.copy(img)

    sx_binary = abs_sobel_thresh(comb_img, sobel_kernel=3, orient='x', thresh=(15, 100))
    s_binary = SChannel_threshold(comb_img, thresh=(120, 255))
    r_binary = RChannel_threshold(comb_img, thresh=(180, 255))
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sx_binary)
    combined_binary[(s_binary == 1) | (sx_binary == 1) | (r_binary == 1)] = 1
    if showresults:
        #Plot the result
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()

        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=40)
        ax2.imshow(combined_binary, cmap='gray')
        ax2.set_title('Combined thresholding result', fontsize=40)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()
        if len(filename) != 0:
            f.savefig(filename, dpi=f.dpi)
    return combined_binary


def warper(src=None, dst=None, inverse=False):
    """Warps the image according to source and destination points."""

    if src is None:
        src = np.float32([[585, 456],
                          [700, 456],
                          [1030, 668],
                          [290, 668]])

    if dst is None:
        dst = np.float32([[300, 70],
                          [1000, 70],
                          [1000, 600],
                          [300, 600]])

    # Given src and dst points, calculate the perspective transform matrix
    if not inverse:
        M = cv2.getPerspectiveTransform(src, dst)
    else:  # This is actually the calculation of the inverse matrix, Minv.
        M = cv2.getPerspectiveTransform(dst, src)
    return M


def warpImage(img, M, showresults=False):
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size)

    if showresults:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(img, cmap='gray')
        ax1.set_title('Original undistored Image', fontsize=40)
        ax2.imshow(warped, cmap='gray')
        ax2.set_title('Warped top-down image', fontsize=40)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()
        f.savefig('../output_images/warped_img.jpg', dpi=f.dpi)
    # Return the resulting image and matrix
    return warped


def quick_detect_lane_lines(image, last_lanes):
    nonzero = image.nonzero()
    nonzero_x, nonzero_y = np.array(nonzero[1]), np.array(nonzero[0])

    last_left_p = np.poly1d(last_lanes.left.pixels.fit)
    last_right_p = np.poly1d(last_lanes.right.pixels.fit)

    margin = 100

    left_lane_indices = ((nonzero_x > (last_left_p(nonzero_y) - margin)) &
                         (nonzero_x < (last_left_p(nonzero_y) + margin)))

    right_lane_indices = ((nonzero_x > (last_right_p(nonzero_y) - margin)) &
                          (nonzero_x < (last_right_p(nonzero_y) + margin)))

    # Again, extract left and right line pixel positions
    left_x = nonzero_x[left_lane_indices]
    left_y = nonzero_y[left_lane_indices]
    right_x = nonzero_x[right_lane_indices]
    right_y = nonzero_y[right_lane_indices]

    left = Lane(left_x, left_y)
    right = Lane(right_x, right_y)

    return Lanes(left, right), image


def full_detect_lane_lines(image):
    # Settings
    window_margin = 100          # This will be +/- on left and right sides of the window
    min_pixels_to_recenter = 50  # Minimum number of pixels before recentering the window
    num_windows = 9             # Number of sliding windows

    image_height, image_width = image.shape

    # Incoming image should already be  undistorted, transformed top-down, and
    # passed through thresholding. Takes histogram of lower half of the image.
    histogram = np.sum(image[image_height//2:,:], axis=0)

    # Placeholder for the image to be returned
    out_image = np.dstack((image, image, image))*255

    # Find peaks on left and right halves of the image
    midpoint = image_width//2
    base_left_x = np.argmax(histogram[:midpoint])
    base_right_x = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows based on num_windows
    window_height = image_height//num_windows

    # Get points of non-zero pixels in image
    nonzero = image.nonzero()
    nonzero_x, nonzero_y = np.array(nonzero[1]), np.array(nonzero[0])

    # Initialize current position, will be updated in each window
    current_left_x = base_left_x
    current_right_x = base_right_x

    # This is where the lane indices will be stored
    left_lane_indices = []
    right_lane_indices = []

    for window in range(num_windows):
        # Get the window boundaries
        window_y_low = image_height - (window + 1) * window_height
        window_y_high = image_height - window * window_height

        window_left_x_low = current_left_x - window_margin
        window_left_x_high = current_left_x + window_margin

        window_right_x_low = current_right_x - window_margin
        window_right_x_high = current_right_x + window_margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_image, (window_left_x_low, window_y_low), (window_left_x_high, window_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_image, (window_right_x_low, window_y_low), (window_right_x_high, window_y_high), (0, 255, 0), 2)

        # Identify the non-zero points within the window
        good_left_indices = ((nonzero_y >= window_y_low) &
                             (nonzero_y < window_y_high) &
                             (nonzero_x >= window_left_x_low) &
                             (nonzero_x < window_left_x_high)).nonzero()[0]

        good_right_indices = ((nonzero_y >= window_y_low) &
                              (nonzero_y < window_y_high) &
                              (nonzero_x >= window_right_x_low) &
                              (nonzero_x < window_right_x_high)).nonzero()[0]

        # Append the indices to the list
        left_lane_indices.append(good_left_indices)
        right_lane_indices.append(good_right_indices)

        if(len(good_left_indices) > min_pixels_to_recenter):
            current_left_x = np.int(np.mean(nonzero_x[good_left_indices]))

        if(len(good_right_indices) > min_pixels_to_recenter):
            current_right_x = np.int(np.mean(nonzero_x[good_right_indices]))

    # Concatenate indices so it becomes a flat array
    left_lane_indices = np.concatenate(left_lane_indices)
    right_lane_indices = np.concatenate(right_lane_indices)

    # Extract the land right lane pixels
    left_x, left_y = nonzero_x[left_lane_indices], nonzero_y[left_lane_indices]
    right_x, right_y = nonzero_x[right_lane_indices], nonzero_y[right_lane_indices]

    left = Lane(left_x, left_y)
    right = Lane(right_x, right_y)

    return Lanes(left, right), out_image


def detect_lane_lines(image, last_lanes=None):
    if (last_lanes is None):
        return full_detect_lane_lines(image )
    else:
        return quick_detect_lane_lines(image, last_lanes)


def display_lanes_window(out_img, lanes):

    # y_max = img.shape[0]
    # Generate x and y values for plotting
    left_fit = lanes.left.pixels.fit
    right_fit = lanes.right.pixels.fit
    left_points = [lanes.left.xs, lanes.left.ys]
    right_points = [lanes.right.xs, lanes.right.ys]
    fity = np.linspace(0, out_img.shape[0] - 1, out_img.shape[0])
    fit_leftx = left_fit[0] * fity ** 2 + left_fit[1] * fity + left_fit[2]
    fit_rightx = right_fit[0] * fity ** 2 + right_fit[1] * fity + right_fit[2]

    out_img[left_points[1], left_points[0]] = [255, 0, 0]
    out_img[right_points[1], right_points[0]] = [0, 0, 255]
    f = plt.figure()
    plt.imshow(out_img)
    plt.plot(fit_leftx, fity, color='yellow')
    plt.plot(fit_rightx, fity, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.title('Detected lane')
    plt.show()
    f.savefig('../output_images/lanes.jpg', dpi=f.dpi, bbox_inches='tight')


class Pipeline:
    def __init__(self):
        dist_pickle = pickle.load(open("../results/wide_dist_pickle.p", "rb"))
        self.mtx = dist_pickle["mtx"]
        self.dist = dist_pickle["dist"]
        self.warp = warper()
        self.unwarp = warper(inverse=True)
        self.last_lanes = None
        self.lanes_average = LanesAverage()
    def process_image(self, image):
        image_height, image_width, _ = image.shape
        undistorted = cal_undistort(image, self.mtx, self.dist)
        warpedImage = warpImage(undistorted, self.warp)
        warpedBinary = combined_threshold(warpedImage)

        lanes, _ = detect_lane_lines(warpedBinary, self.last_lanes)
        self.lanes_average.update(lanes)
        if self.last_lanes is None:
            self.last_lanes = lanes
        if lanes.lanes_parallel(image_height) and lanes.distance_from_center((image_width/2, image_height)) < 4.0:
            self.last_lanes = lanes
        return overlay_detected_lane_data(image, self.lanes_average, self.unwarp)

if __name__ == '__main__':
    dist_pickle = pickle.load(open("../results/wide_dist_pickle.p", "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    img = cv2.imread('../camera_cal/calibration1.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    M = warper()
    invM = warper(inverse=True)
    undistorted = cal_undistort(img, mtx, dist, True, '../output_images/undist_chess.jpg')
    filename = 'straight_lines2.jpg'
    image = mpimg.imread('../test_images/' + filename)
    undistorted = cal_undistort(image, mtx, dist)
    warpedImage = warpImage(undistorted, M, True)
    filename = 'test2.jpg'
    image = mpimg.imread('../test_images/' + filename)
    undistorted = cal_undistort(image, mtx, dist, True, '../output_images/undist_img.jpg')
    combined_threshold(undistorted, True, '../output_images/binarize_unwarp_comb.jpg')
    warpedImage= warpImage(undistorted, M)
    warpedBinary = combined_threshold(warpedImage, True, '../output_images/binarize_warp_comb.jpg')
    lanes, out = detect_lane_lines(warpedBinary)
    display_lanes_window(out, lanes)
    img = overlay_detected_lane_data(image, lanes, invM, True)

    np.seterr(all='ignore')
    pipeline = Pipeline()
    clip1 = VideoFileClip("../video/project_video.mp4", audio=False)
    white_clip = clip1.fl_image(pipeline.process_image)
    white_clip.write_videofile("../output_images/project_video.mp4", audio=False)
