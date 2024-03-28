import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt
from matplotlib import patches
from scipy.linalg import qr

path_left_images = "/Users/nitishsanghi/Documents/StereoMatching/data_scene_flow/training/image_2/"
path_right_images = "/Users/nitishsanghi/Documents/StereoMatching/data_scene_flow/training/image_3"
path_cam_cals = "/Users/nitishsanghi/Documents/StereoMatching/data_scene_flow_calib/training/calib_cam_to_cam"

left_image_files = glob.glob(path_left_images+'/*')
right_image_files = glob.glob(path_right_images+'/*')

baseline = 0.54 # camera center to camera center distance in meters

cam_cal_files = glob.glob(path_cam_cals + '/*')

cam_cals_file = cam_cal_files[0]
params_dict = {}
with open(cam_cals_file) as file:
    for line in file:
        params = line.split(" ")
        params_dict[params[0]] = params[1:]

def list2float(params):
    return list(map(float, params))

def camera_params(params_dict):

    params = {}
    #Image size
    params['raw_image_size'] = np.array([[int(float(params_dict['S_02:'][0]))], [int(float(params_dict['S_02:'][1]))]])
    params['rect_image_size'] = np.array([[int(float(params_dict['S_rect_02:'][0]))], [int(float(params_dict['S_rect_02:'][1]))]])
    
    #Camera Instrinsics
    params['left_cam_intrinsic'] = np.array([list2float(params_dict['K_02:'][:3]), list2float(params_dict['K_02:'][3:6]), list2float(params_dict['K_02:'][6:])])
    params['right_cam_intrinsic'] = np.array([list2float(params_dict['K_03:'][:3]), list2float(params_dict['K_03:'][3:6]), list2float(params_dict['K_03:'][6:])])
    
    #Camera Distortions
    params['left_dist'] = np.array(list2float(params_dict['D_02:']))
    params['right_dist'] = np.array(list2float(params_dict['D_03:']))

    #Camera Raw Rotation Extrinsic
    params['left_raw_rot_ext'] = np.array([list2float(params_dict['R_02:'][:3]), list2float(params_dict['R_02:'][3:6]), list2float(params_dict['R_02:'][6:])])
    params['right_raw_rot_ext'] = np.array([list2float(params_dict['R_03:'][:3]), list2float(params_dict['R_03:'][3:6]), list2float(params_dict['R_03:'][6:])])

    #Camera Raw Translation Extrinsic
    params['left_raw_trans_ext'] = np.array([list2float(params_dict['T_02:'][:1]), list2float(params_dict['T_02:'][1:2]), list2float(params_dict['T_02:'][2:])])
    params['right_raw_trans_ext'] = np.array([list2float(params_dict['T_03:'][:1]), list2float(params_dict['T_03:'][1:2]), list2float(params_dict['T_03:'][2:])])

    #Camera Rect Rotation Extrinsic
    params['left_rect_rot_ext'] = np.array([list2float(params_dict['R_rect_02:'][:3]), list2float(params_dict['R_rect_02:'][3:6]), list2float(params_dict['R_rect_02:'][6:])])
    params['right_rect_rot_ext'] = np.array([list2float(params_dict['R_rect_03:'][:3]), list2float(params_dict['R_rect_03:'][3:6]), list2float(params_dict['R_rect_03:'][6:])])

    #Camera Projection 
    params['left_rect_proj'] = np.array([list2float(params_dict['P_rect_02:'][:4]), list2float(params_dict['P_rect_02:'][4:8]), list2float(params_dict['P_rect_02:'][8:])])
    params['right_rect_proj'] = np.array([list2float(params_dict['P_rect_03:'][:4]), list2float(params_dict['P_rect_03:'][4:8]), list2float(params_dict['P_rect_03:'][8:])])

    return params

def RQ_Decomposition(projection):

    M = projection[:,:3]
    reversed_M = np.flipud(M)
    Q, R = qr(np.transpose(reversed_M))

    R = np.flipud(np.transpose(R))
    R = np.fliplr(R)

    Q = np.transpose(Q)
    Q = np.flipud(Q)

    return R, Q

def projection_decomp(projection):
    R, Q = RQ_Decomposition(projection)
    t = np.matmul(-np.linalg.inv(R),projection[:,3])
    return R, Q, t

params = camera_params(params_dict)

K_left, R_left, t_left = projection_decomp(params['left_rect_proj'])
K_right, R_right, t_right = projection_decomp(params['right_rect_proj'])

left_image = cv2.imread('/Users/nitishsanghi/Documents/StereoMatching/data_scene_flow/training/image_2/000000_10.png')
right_image = cv2.imread('/Users/nitishsanghi/Documents/StereoMatching/data_scene_flow/training/image_3/000000_10.png')

def add_padding(image, window_size):
    pad = window_size//2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    return image

def block_indices(row, col, window_size):
    row_min = col - window_size//2 - 1
    row_max = col + window_size//2
    col_min = row - window_size//2 - 1
    col_max = row + window_size//2
    return row_min, col_min, row_max, col_max

def num_sliding_windows(n_cols, col, num_disparity_range, window_size, disparity_type):
    num_window_reducer = 0

    if disparity_type == 'left':
        if col - num_disparity_range - window_size//2 <= 0:
            num_window_reducer = num_disparity_range - col + window_size//2

    if disparity_type == 'right':
        if n_cols - col - num_disparity_range <= 0:
            num_window_reducer = num_disparity_range + col - n_cols
    
    sliding_windows = num_disparity_range - num_window_reducer
    return sliding_windows

def ref_window_stacker(window, sliding_windows):
    window_stack = np.stack([window for n in range(sliding_windows)], axis=2)
    return window_stack

def target_window_stacker(target_image_aug, sliding_windows, window_coords, disparity_type):
    sign = 1 if disparity_type == 'left' else -1
    window_stack = [target_image_aug[window_coords[1] : window_coords[3], window_coords[0] - sign*d : window_coords[2] - sign*d] for d in range(sliding_windows)]
    window_stack= np.stack(window_stack, axis=2)
    return window_stack

def min_disparity_cost_ssd(ref_window_stack, target_window_stack):
    return np.argmin(np.sum(np.square(ref_window_stack - target_window_stack),(0,1)))

def stack_diff_ssd(window_stack):
    mean = np.mean(window_stack,(0,1))
    diff = window_stack - mean
    ssd = np.sum(np.square(diff),(0,1))
    return mean, ssd

def min_disparity_cost_ncc(ref_window_stack, target_window_stack):
    ref_mean = np.mean(ref_window_stack,(0,1))
    target_mean = np.mean(target_window_stack,(0,1))
    
    ref_diff = ref_window_stack - ref_mean
    target_diff = target_window_stack - target_mean
    numerator = np.sum(np.multiply(ref_diff, target_diff),(0,1))
    
    ref_ssd = np.sum(np.square(ref_diff),(0,1))
    target_ssd = np.sum(np.square(target_diff),(0,1))
    denominator = np.multiply(ref_ssd, target_ssd)
    epsilon = 1e-8 
    denominator = np.where(denominator < epsilon, epsilon, denominator)
    return np.argmin(numerator/denominator)

def convert_images_grayscale(ref_image, target_image):
    return cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY), cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

def add_padding_images(ref_image, target_image, window_size):
    return add_padding(ref_image, window_size), add_padding(target_image, window_size)

def stereo_match_ssd_new(ref_image, target_image, disparity_type, similarity_metrics, window_size = 11, num_disparity_range = 100):

    ref_image, target_image = convert_images_grayscale(ref_image, target_image)

    ref_image_aug, target_image_aug = add_padding_images(ref_image, target_image, window_size)

    ref_disparity = np.zeros(ref_image.shape)

    n_rows, n_cols = ref_image.shape
    for row in range(window_size//2+1, n_rows):
        for col in range(window_size//2+1, n_cols):

            window_coords = block_indices(row, col, window_size)

            ref_window = ref_image_aug[window_coords[1] : window_coords[3], window_coords[0] : window_coords[2]]
            
            sliding_windows = num_sliding_windows(n_cols, col, num_disparity_range, window_size, disparity_type)

            ref_window_stack = ref_window_stacker(ref_window, sliding_windows)

            target_window_stack= target_window_stacker(target_image_aug, sliding_windows, window_coords, disparity_type)
            #min_disparity_cost_ncc(ref_window_stack, target_window_stack)
            ref_disparity[row,col] = similarity_metrics(ref_window_stack, target_window_stack)

    return ref_disparity

left_ssd = stereo_match_ssd_new(left_image, right_image, 'left', min_disparity_cost_ssd)
right_ssd = stereo_match_ssd_new(right_image, left_image, 'right', min_disparity_cost_ssd)
left_ncc = stereo_match_ssd_new(left_image, right_image, 'left', min_disparity_cost_ncc)
right_ncc = stereo_match_ssd_new(right_image, left_image, 'right', min_disparity_cost_ncc)

def depth_map(image, baseline, focal):
    image[image <= 0] = 0.1
    return focal*baseline/image


cv2.imwrite("left_ssd.jpg",left_ssd)
cv2.imwrite("left_ncc.jpg",left_ncc)
cv2.imwrite("right_ssd.jpg",right_ssd)
cv2.imwrite("right_ncc.jpg",right_ncc)


#depth_l = depth_map(left, .54, 722)
#depth_r = depth_map(left, .54, 722)

#_, ax = plt.subplots(1,2)
#plt.imshow(left_image)
#plt.imshow(right_image)
#plt.imshow(left)
#plt.show()
#plt.imshow(right)
#plt.imshow(left - right)
#plt.imshow(depth_l)
#plt.imshow(depth_r)
plt.show()