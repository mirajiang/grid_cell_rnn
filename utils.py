import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy
import scipy.ndimage

def generate_run_ID(options):
    ''' 
    Create a unique run ID from the most relevant
    parameters.
    '''

    params = [
        'steps', str(options.sequence_length),
        'batch', str(options.batch_size),
        str(options.activation),
        'gridcell', str(options.Ng),
        'rf', str(options.place_cell_rf),
        'DoG', str(options.DoG),
        'periodic', str(options.periodic),
        'lr', str(options.learning_rate),
        'weight_decay', str(options.weight_decay),
        'vsgm', str(options.vel_sigma),
        'vscl', str(options.vel_scale),
        'hsgm', str(options.hid_sigma),
        'hscl', str(options.hid_scale),
        options.replication_num,
        #'seed', str(options.seed),
        ]
    separator = '_'
    run_ID = separator.join(params)
    run_ID = run_ID.replace('.', '')

    return run_ID

# mostly copied from https://github.com/ganguli-lab/grid-pattern-formation/blob/master/visualize.py
# functions for plotting grid cells
def compute_ratemaps(model, trajectory_generator, options, res=20, n_avg=None, Ng=512, idxs=None):
    '''Compute spatial firing fields'''

    if not n_avg:
        n_avg = 1000    # number of times to sample trajectories to to compute grid cell field

    if not np.any(idxs):
        idxs = np.arange(Ng)
    idxs = idxs[:Ng]

    g = np.zeros([n_avg, options.batch_size * options.sequence_length, Ng])
    pos = np.zeros([n_avg, options.batch_size * options.sequence_length, 2])

    activations = np.zeros([Ng, res, res])
    counts = np.zeros([res, res])

    for index in range(n_avg):
        inputs, pos_batch, _ = trajectory_generator.get_test_batch(batch_size=options.batch_size)
        g_batch = model.grid(inputs, store_neurons=True).detach().cpu().numpy()
        
        pos_batch = np.reshape(pos_batch.cpu(), [-1, 2])
        g_batch = g_batch[:,:,idxs].reshape(-1, Ng)
        
        g[index] = g_batch
        pos[index] = pos_batch

        x_batch = (pos_batch[:,0] + options.box_width/2) / (options.box_width) * res
        y_batch = (pos_batch[:,1] + options.box_height/2) / (options.box_height) * res
    
        for i in range(options.batch_size*options.sequence_length):
            x = x_batch[i]
            y = y_batch[i]
            if x >=0 and x < res and y >=0 and y < res:
                counts[int(x), int(y)] += 1
                activations[:, int(x), int(y)] += g_batch[i, :]


    for x in range(res):
        for y in range(res):
            if counts[x, y] > 0:
                activations[:, x, y] /= counts[x, y]
                
    g = g.reshape([-1, Ng])
    pos = pos.reshape([-1, 2])

    # # scipy binned_statistic_2d is slightly slower
    # activations = scipy.stats.binned_statistic_2d(pos[:,0], pos[:,1], g.T, bins=res)[0]
    rate_map = activations.reshape(Ng, -1)

    return activations, rate_map, g, pos, counts

def concat_images(images, image_width, spacer_size):
    """ Concat image horizontally with spacer """
    spacer = np.ones([image_width, spacer_size, 4], dtype=np.uint8) * 255
    images_with_spacers = []

    image_size = len(images)

    for i in range(image_size):
        images_with_spacers.append(images[i])
        if i != image_size - 1:
            # Add spacer
            images_with_spacers.append(spacer)
    ret = np.hstack(images_with_spacers)
    return ret


def concat_images_in_rows(images, row_size, image_width, spacer_size=4):
    '''Concat images in rows '''
    column_size = len(images) // row_size
    spacer_h = np.ones([spacer_size, image_width*column_size + (column_size-1)*spacer_size, 4],
                       dtype=np.uint8) * 255

    row_images_with_spacers = []

    for row in range(row_size):
        row_images = images[column_size*row:column_size*row+column_size]
        row_concated_images = concat_images(row_images, image_width, spacer_size)
        row_images_with_spacers.append(row_concated_images)

        if row != row_size-1:
            row_images_with_spacers.append(spacer_h)

    ret = np.vstack(row_images_with_spacers)
    return ret


def convert_to_colormap(im, cmap):
    im = cmap(im)
    im = np.uint8(im * 255)
    return im


def rgb(im, cmap='jet', smooth=True):
    cmap = plt.cm.get_cmap(cmap)
    np.seterr(invalid='ignore')  # ignore divide by zero err
    im = (im - np.min(im)) / (np.max(im) - np.min(im))
    if smooth:
        im = cv2.GaussianBlur(im, (3,3), sigmaX=1, sigmaY=0)
    im = cmap(im)
    im = np.uint8(im * 255)
    return im


def plot_ratemaps(activations, n_plots, cmap='jet', smooth=True, width=16):
    images = [rgb(im, cmap, smooth) for im in activations[:n_plots]]
    rm_fig = concat_images_in_rows(images, n_plots//width, activations.shape[-1])
    return rm_fig

def plot_grid_cells(options,model,trajectory_generator,res,n_avg,perturbation=None):
    Ng = options.Ng
    activations, _, _, _, counts = compute_ratemaps(model,trajectory_generator,options,res=res,n_avg=n_avg,Ng=Ng)

    n_plot = 256
    plt.figure(figsize=(16,4*n_plot//8**2))
    rm_fig = plot_ratemaps(activations, n_plot, smooth=True)
    print(rm_fig.shape)
    plt.imshow(rm_fig)
    plt.axis('off')
    #plt.savefig(f"images/{options.run_ID}/{generate_dir_name(options,perturbation)}/grid_cell.pdf")
    #plt.clf()
    #plt.close()

    return activations


# Functions for computing grid scores and tracking translational and rotational drift across epochs
def pearson_clear_0 (arr1, arr2):
    '''
    Helper function to clear out 0s from a mask and return the pearson correlation between rotated and original autocorrelation arrays (used in calc_grid)
    '''
    mask = (arr1 != 0) & (arr2 != 0)
    if np.all(~mask):
        return 0
    return scipy.stats.pearsonr(arr1[mask], arr2[mask])[0] # Returns correlation

def get_radius(arr):
    '''
    Given an array, get the radial distance of each point to the center of the array
    '''
    # Get origin
    center_idx = [int(arr.shape[0] / 2), int(arr.shape[1] / 2)]
    x_shape = arr[0].shape[0]
    y_shape = arr[1].shape[0]

    # Get x and y coordinates
    x, y = np.meshgrid(np.linspace(center_idx[0] - x_shape + 1, x_shape - center_idx[0] - 1, x_shape), np.linspace(center_idx[1] - y_shape + 1, y_shape - center_idx[1] - 1, y_shape))

    # Convert x and y to polar
    return np.sqrt(x**2 + y**2)

def mask(autocorr, rad, n_bins=50):
    '''
    Get mask for a grid cell ratemap

    Args:
        autocorr: 2D autocorrelagram for a grid cell ratemap ([res * 2, res * 2]).
        rad: 2D array of radial coordinates that give the distance of point of the autocorrelagram from the 2D center of the array (rad.shape = autocorr.shape).

    Returns:
        masked_im: circular mask of the autocorrelagram, excluding the central peak and the area past the first ring of peaks
    '''
    flattened_corr = autocorr.flatten()

    r = rad.flatten()
    min_r = np.min(r)
    max_r = np.max(r)
    bin_size = max_r / n_bins
    bins = np.arange(min_r, max_r, bin_size)
    bin_indices = np.ceil(r / bin_size).astype(int)

    autocorr_radial = np.bincount(bin_indices, weights = flattened_corr, minlength = n_bins) / np.bincount(bin_indices, minlength = n_bins)
    smooth_ac = scipy.ndimage.gaussian_filter(autocorr_radial, 0.7)

    peak_idx = (int)(autocorr_radial.shape[0] * 0.25)

    # Extract the first peak
    for i in range(1, smooth_ac.shape[0] - 1):
        if (smooth_ac[i + 1] < smooth_ac[i] > smooth_ac[i - 1]):
            peak_idx = i
            break

    peak = peak_idx * bin_size
    r0 = peak * 0.5
    r1 = peak * 1.5

    # Mask
    upper_mask = rad < r1
    upper_lim = np.where(upper_mask, autocorr, 0)

    lower_mask = rad > r0
    masked_im = np.where(lower_mask, upper_lim, 0)

    return masked_im

def calc_grid(map):
    '''
    Given a map of activations for a grid cell, compute the grid score
    '''

    autocorr = scipy.signal.correlate2d(map, map)
    flattened_corr = autocorr.flatten()

    # Convert x and y to polar
    r = get_radius(autocorr)
    
    # Get origin
    center_idx = [int(autocorr.shape[0] / 2), int(autocorr.shape[1] / 2)]
    x_shape = autocorr[0].shape[0]
    y_shape = autocorr[1].shape[0]

    # Get x and y coordinates
    x, y = np.meshgrid(np.linspace(center_idx[0] - x_shape + 1, x_shape - center_idx[0] - 1, x_shape), np.linspace(center_idx[1] - y_shape + 1, y_shape - center_idx[1] - 1, y_shape))

    # Convert x and y to polar
    rad = np.sqrt(x**2 + y**2)
    r = rad.flatten()
    
    # To bin
    n_bins = 50
    min_r = np.min(r)
    max_r = np.max(r)
    bin_size = max_r / n_bins
    bins = np.arange(min_r, max_r, bin_size)
    bin_indices = np.ceil(r / bin_size).astype(int)

    autocorr_radial = np.bincount(bin_indices, weights = flattened_corr, minlength = n_bins) / np.bincount(bin_indices, minlength = n_bins)
    smooth_ac = scipy.ndimage.gaussian_filter(autocorr_radial, 0.7)

    peak_idx = (int)(autocorr_radial.shape[0] * 0.25)

    # Extract the first peak
    for i in range(1, smooth_ac.shape[0] - 1):
        if (smooth_ac[i + 1] < smooth_ac[i] > smooth_ac[i - 1]):
            peak_idx = i
            break

    peak = peak_idx * bin_size
    r0 = peak * 0.5
    r1 = peak * 1.5

    # Mask
    upper_mask = rad < r1
    upper_lim = np.where(upper_mask, autocorr, 0)

    lower_mask = rad > r0
    masked_im = np.where(lower_mask, upper_lim, 0)

    # Rotate the disk and compute the corresponding correlations for the grid score
    autocorr_flat = masked_im.flatten()

    rot_60 = scipy.ndimage.rotate(masked_im, 60, reshape=False).flatten()
    corr_60 = pearson_clear_0(autocorr_flat, rot_60)

    rot_120 = scipy.ndimage.rotate(masked_im, 120, reshape=False).flatten()
    corr_120 = pearson_clear_0(autocorr_flat, rot_120)

    rot_30 = scipy.ndimage.rotate(masked_im, 30, reshape=False).flatten()
    corr_30 = pearson_clear_0(autocorr_flat, rot_30)

    rot_90 = scipy.ndimage.rotate(masked_im, 90, reshape=False).flatten()
    corr_90 = pearson_clear_0(autocorr_flat, rot_90)

    rot_150 = scipy.ndimage.rotate(masked_im, 150, reshape=False).flatten()
    corr_150 = pearson_clear_0(autocorr_flat, rot_150)

    grid_score = (corr_60 + corr_120) / 2  - (corr_30 + corr_90 + corr_150) / 3

    return grid_score


def translational_drift(grid_activations, n_plots, res, height, width, epoch1, epoch2):
    '''
    Calculate translational drift of grid cells between two epochs (height and width given in meters)
    '''
    displacements = np.zeros(n_plots)
    x = np.zeros(n_plots)
    y = np.zeros(n_plots)

    # Conversion factor from bin units to centimeters
    x_factor = (width * 100) / res
    y_factor = (height * 100) / res

    for i in range(0, n_plots):
        cross_corr = scipy.signal.fftconvolve(grid_activations[epoch1, i], np.flip(grid_activations[epoch2, i], axis = (0, 1)), mode='same')
        center = (int(cross_corr[0].shape[0] / 2), int(cross_corr[1].shape[0] / 2))

        # Smoothing the cross-correlogram
        cross_corr = scipy.ndimage.gaussian_filter(cross_corr, 5)

        # Getting the center part of the cross-correlogram
        cross_corr_center = cross_corr[center[0] - 10:center[0] + 10, center[1] - 10:center[1] + 10]
        new_center = (int(cross_corr_center[0].shape[0] / 2), int(cross_corr_center[1].shape[0] / 2))

        peak = np.unravel_index(np.argmax(cross_corr_center), cross_corr_center.shape)

        # Get changes in x and changes in y
        x[i] = (peak[0] - new_center[0]) * x_factor
        y[i] = (peak[1] - new_center[1]) * y_factor

        # Get distance of the peak to the center
        displacement = np.sqrt(((new_center[0] - peak[0]) * x_factor)**2 + ((new_center[1] - peak[1]) * y_factor)**2)
        displacements[i] = displacement;

    return np.mean(displacements), displacements, np.mean(x), np.mean(y)

def rotational_drift(grid_activations, n_plots, epoch1, epoch2):
    '''
    Calculate rotational drift of grid cells between two epochs
    '''
    rotations = np.zeros(n_plots)

    for i in range(n_plots):
        # Get the autocorrelograms of the grids to compare
        autocorr1 = scipy.signal.fftconvolve(grid_activations[epoch1, i], np.flip(grid_activations[epoch1, i], axis=(0, 1)), mode='same')
        autocorr2 = scipy.signal.fftconvolve(grid_activations[epoch2, i], np.flip(grid_activations[epoch2, i], axis=(0, 1)), mode='same')

        # Apply smoothing filters
        autocorr1 = scipy.ndimage.gaussian_filter(autocorr1, 5)
        autocorr2 = scipy.ndimage.gaussian_filter(autocorr2, 5)

        # Get radial coordinates
        rad = get_radius(autocorr1)

        # Masking
        masked_im1 = mask(autocorr1, rad)
        masked_im2 = mask(autocorr2, rad).flatten()

        # Rotate and compute the pearson cross-correlation
        rot_crosscorr = np.zeros(60)
        for deg in range(60):
            rot_im = scipy.ndimage.rotate(masked_im1, deg, reshape=False).flatten()
            rot_crosscorr[deg] = pearson_clear_0(masked_im2, rot_im)

        # Get the first peak of the cross-correlation (should correspond to the rotation)
        for x in range(1, rot_crosscorr.shape[0] - 1):
            if (rot_crosscorr[x + 1] < rot_crosscorr[x] > rot_crosscorr[x - 1]):
                # Accounting for rotations in the opposite direction
                if x > 29:
                    rotations[i] = 60 - x
                    break
                rotations[i] = x
                break
    
    return np.mean(rotations)
