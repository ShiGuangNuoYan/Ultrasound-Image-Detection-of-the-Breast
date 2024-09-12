import numpy as np
import numpy.typing as npt
from tqdm import tqdm
def _normal_gaussian_fuzzy_membership(
    std: int, a: npt.NDArray[np.int32], b: float
) -> npt.NDArray[np.float_]:

    precomputed_sigma: int = 2 * np.square(std) + 10e-5

    divider: float = 1 / np.sqrt(precomputed_sigma * np.pi)
    exponential: npt.NDArray[np.float_] = np.exp(-np.square(a - b) / precomputed_sigma)
    delta_x: npt.NDArray[np.float_] = divider * exponential
    return delta_x


def _window_membership(image_window: npt.NDArray[np.int32]) -> npt.ArrayLike:
    i_avg: float = np.mean(image_window)
    std: float = np.std(image_window)

    # Calculate fuzzy membership
    membership = _normal_gaussian_fuzzy_membership(std, image_window, i_avg)
    # Get the 2D coordinate of the max value
    row_max, col_max = np.unravel_index(membership.argmax(), membership.shape)
    # Translate the coord from membership to the image window
    image_val_max_member = image_window[row_max, col_max]

    return image_val_max_member


def gaussian_fuzzy_filter(
     image_array: npt.NDArray[np.int32], width: int = 3
):
    # Width should be odd
    if width % 2 != 1:
        raise Exception("Window width needs to be odd")

    padded_image: npt.NDArray[np.int32] = np.pad(image_array, (width - 1) // 2)
    image_array_processed = np.copy(image_array)
    n_row, n_col = image_array.shape
    # For loop is unoptimizable due to the fact that the convolution process is sequential
    for i_row in tqdm(range(n_row)):
        for i_col in range(n_col):
            image_array_processed[i_row, i_col] = _window_membership(
                padded_image[i_row : i_row + width, i_col : i_col + width]
            )
    return image_array_processed


def psnr(original, filtered):
    mse = np.mean((original - filtered) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr