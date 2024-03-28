import cv2
import time
import numpy as np

class StereoMatcher:
    def __init__(self, ref_image, target_image, window_size=11, num_disparity_range=100):
        self.ref_image = ref_image
        self.target_image = target_image
        self.window_size = window_size
        self.pad = self.window_size // 2
        self.num_disparity_range = num_disparity_range

        # Convert images to grayscale and add padding
        self.ref_image_gray, self.target_image_gray = self._convert_images_grayscale()
        self.ref_image_padded, self.target_image_padded = self._add_padding_images()

    def _convert_images_grayscale(self):
        return cv2.cvtColor(self.ref_image, cv2.COLOR_BGR2GRAY), cv2.cvtColor(self.target_image, cv2.COLOR_BGR2GRAY)

    def _add_padding(self, image):
        return cv2.copyMakeBorder(image, self.pad, self.pad, self.pad, self.pad, cv2.BORDER_REFLECT)

    def _add_padding_images(self):
        return self._add_padding(self.ref_image_gray), self._add_padding(self.target_image_gray)

    def _block_indices(self, row, col):
        row_min = col - self.pad - 1
        row_max = col + self.pad
        col_min = row - self.pad - 1
        col_max = row + self.pad
        return row_min, col_min, row_max, col_max

    def _num_sliding_windows(self, n_cols, col, disparity_type):
        num_window_reducer = 0

        if disparity_type == 'left':
            if col - self.num_disparity_range - self.pad <= 0:
                num_window_reducer = self.num_disparity_range - col + self.pad

        if disparity_type == 'right':
            if n_cols - col - self.num_disparity_range <= 0:
                num_window_reducer = self.num_disparity_range + col - n_cols

        sliding_windows = self.num_disparity_range - num_window_reducer
        return sliding_windows

    def _ref_window_stacker(self, window, sliding_windows):
        return  np.stack([window] * sliding_windows, axis=2)

    def _target_window_stacker(self, target_image_aug, sliding_windows, window_coords, disparity_type):
        sign = 1 if disparity_type == 'left' else -1
        return np.stack([target_image_aug[window_coords[1]:window_coords[3], window_coords[0] - sign * d: window_coords[2] - sign * d] for d in range(sliding_windows)], axis=2)

    def _min_disparity_cost_ssd(self, ref_window_stack, target_window_stack):
        return np.argmin(np.sum(np.square(ref_window_stack - target_window_stack), (0, 1)))

    def _stack_zncc_mean_ssd(self, window_stack):
        mean = np.mean(window_stack, (0, 1))
        diff = window_stack - mean
        ssd = np.sum(np.square(diff), (0, 1))
        return mean, ssd
    
    def _stack_cc_mean_ssd(self, window_stack):
        mean = np.mean(window_stack, (0, 1))
        ssd = np.sum(np.square(window_stack), (0, 1))
        return mean, ssd

    def _min_disparity_cost_zncc(self, ref_window_stack, target_window_stack):
        ref_mean, ref_ssd = self._stack_zncc_mean_ssd(ref_window_stack)
        target_mean, target_ssd = self._stack_zncc_mean_ssd(target_window_stack)

        # Compute the cross-correlation
        cross_correlation = np.sum((ref_window_stack - ref_mean) * (target_window_stack - target_mean), axis=(0, 1))

        # Compute ZNCC
        epsilon = 1e-10
        zncc = cross_correlation / (np.sqrt(ref_ssd * target_ssd) + epsilon)

        # Find disparity with maximum ZNCC
        return np.argmax(zncc)

    def _min_disparity_cost_cc(self, ref_window_stack, target_window_stack):
        ref_mean, ref_ssd = self._stack_cc_mean_ssd(ref_window_stack)
        target_mean, target_ssd = self._stack_cc_mean_ssd(target_window_stack)

        # Compute the cross-correlation
        cross_correlation = np.sum((ref_window_stack - ref_mean) * (target_window_stack - target_mean), axis=(0, 1))

        # Compute CC
        epsilon = 1e-10
        cc = cross_correlation / (np.sqrt(ref_ssd * target_ssd) + epsilon)

        # Find disparity with maximum CC
        return np.argmax(cc)


    def match(self, disparity_type='left', similarity_metric='ssd'):
        start_time = time.time()
        ref_disparity = np.zeros_like(self.ref_image_gray)

        n_rows, n_cols = self.ref_image_gray.shape
        for row in range(self.pad + 1, n_rows):
            for col in range(self.pad + 1, n_cols):
                sliding_windows = self._num_sliding_windows(n_cols, col, disparity_type)
                window_coords = self._block_indices(row, col)

                ref_window = self.ref_image_padded[window_coords[1]:window_coords[3],
                             window_coords[0]:window_coords[2]]

                ref_window_stack = self._ref_window_stacker(ref_window, sliding_windows)

                target_window_stack = self._target_window_stacker(self.target_image_padded, sliding_windows,
                                                                  window_coords, disparity_type)

                if similarity_metric == 'sad':
                    disparity = np.argmin(np.sum(np.abs(ref_window_stack - target_window_stack),(0,1)))
                elif similarity_metric == 'ssd':
                    disparity = np.argmin(np.sum(np.square(ref_window_stack - target_window_stack), (0, 1)))
                elif similarity_metric == 'cc':
                    disparity = self._min_disparity_cost_cc(ref_window_stack, target_window_stack)
                elif similarity_metric == 'zncc':
                    disparity = self._min_disparity_cost_zncc(ref_window_stack, target_window_stack)
                else:
                    raise ValueError("Invalid similarity metric")

                ref_disparity[row, col] = disparity
        end_time = time.time()
        print(f"Execution time of match method: {end_time - start_time} seconds")

        return ref_disparity


left_image = cv2.imread('/Users/nitishsanghi/Documents/StereoMatching/data_scene_flow/training/image_2/000000_10.png')
right_image = cv2.imread('/Users/nitishsanghi/Documents/StereoMatching/data_scene_flow/training/image_3/000000_10.png')
matcher = StereoMatcher(left_image, right_image, window_size=11, num_disparity_range=100)
left_ssd = matcher.match()
left_zncc = matcher.match(similarity_metric='zncc')
left_sad = matcher.match(similarity_metric='sad')
left_cc = matcher.match(similarity_metric='cc')

cv2.imwrite("left_ssd_v2.jpg",left_ssd)
cv2.imwrite("left_zncc_v2.jpg",left_zncc)
cv2.imwrite("left_sad_v2.jpg",left_sad)
cv2.imwrite("left_cc_v2.jpg",left_cc)