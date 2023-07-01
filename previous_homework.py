from typing import List, Tuple
import numpy as np


def apply_filter(image_array: np.ndarray, kernel: np.ndarray, padding: List[List[int]]) -> np.ndarray:
    """ Apply a filter with the given kernel to the zero padded gray scaled (2D) input image array.
        **Note:** Kernels can be rectangular.
        **Note:** You can use ```np.meshgrid``` and indexing to avoid using loops for convolving.
        **Do not** use ```np.convolve``` in this question.
        **Do not** use ```np.pad```. Use index assignment and slicing with numpy and do not loop
            over the pixels for padding.

    Args:
        image_array (np.ndarray): 2D Input array
        kernel (np.ndarray): 2D kernel array of odd edge sizes
        padding: (List[list[int]]): List of zero paddings. Example: [[3, 2], [1, 4]]. The first list
            [3, 3] determines the padding for the width of the image while [1, 4] determines the
            padding to apply to top and bottom of the image. The resulting image will have a shape
            of ((1 + H + 4), (3 + W + 2)).

    Raises:
        ValueError: If the length of kernel edges are not odd

    Returns:
        np.ndarray: Filtered array (May contain negative values)
    """
    img = image_array

    def __check_kernel():
        res = np.asarray(kernel.shape) % 2 == 0
        if True in res:
            raise ValueError("Kernel edges are not odd.")

    def __pad_image():

        pad_top, pad_bottom = padding[1]
        pad_left, pad_right = padding[0]

        padded_img = np.zeros((img.shape[0]+pad_top+pad_bottom, img.shape[1]+pad_left+pad_right))
        padded_img[pad_top:pad_top+img.shape[0], pad_left:pad_left+img.shape[1]] = img

        return padded_img

    __check_kernel()
    __pad_image()

    step_horizontal = img.shape[0] - kernel.shape[0] + 1  #yatay
    step_vertical = img.shape[1] - kernel.shape[1] + 1    #dikey

    final_img = np.zeros((step_horizontal,step_vertical))

    for i in range(step_horizontal):
        for j in range(step_vertical):
            sub_img = img[i:i+kernel.shape[0], j:j+kernel.shape[1]]
            final_img[i][j] = round(np.sum(np.multiply(sub_img,kernel)))

    return final_img


def gaussian_filter(image_arr: np.ndarray, kernel_size: Tuple[int, int], sigma: float) -> np.ndarray:
    """ Apply Gauss filter that is centered and has the shared standard deviation ```sigma```
    **Note:** Remember to normalize kernel before applying.
    **Note:** You can use ```np.meshgrid``` (once again) to generate Gaussian kernels

    Args:
        image_arr (np.ndarray): 2D Input array of shape (H, W)
        kernel_size (Tuple[int]): 2D kernel size (H, W)
        sigma (float): Standard deviation

    Returns:
        ImageType: Filtered Image array
    """
    def __construct_kernel():
        kernel = np.zeros(kernel_size)
        for i in range(kernel_size[0]):
            for j in range(kernel_size[1]):
                kernel[i][j] = np.exp(-0.5*((i-kernel_size[0]//2)**2+(j-kernel_size[1]//2)**2)/(sigma**2))
        return kernel / np.sum(kernel)

    def __find_padding():
        pad_horizontal = kernel_size[1] // 2
        pad_vertical = kernel_size[0] // 2
        return [[pad_horizontal,pad_horizontal],[pad_vertical,pad_vertical]]

    padding = __find_padding()
    g_kernel = __construct_kernel()

    new_img = apply_filter(image_arr, g_kernel, padding)
    return new_img


def sobel_vertical(image_array: np.ndarray) -> np.ndarray:
    """ Return the output of the vertical Sobel operator with same padding.
        **Note**: This function may return negative values

    Args:
        image_array (np.ndarray): 2D Input array of shape (H, W)

    Returns:
        np.ndarray: Derivative array of shape (H, W).
    """
    img = image_array

    padding = [[0,0],[0,0]]
    kernel = np.asarray([[-1,0,1],[-2,0,2],[-1,0,1]])

    new_img = apply_filter(img, kernel, padding)
    pad_x = (img.shape[0] - new_img.shape[0]) // 2
    pad_y = (img.shape[1] - new_img.shape[1]) // 2

    final_img = np.zeros(img.shape)

    final_img[pad_x:pad_x+new_img.shape[0], pad_y:pad_y+new_img.shape[1]] = new_img

    return final_img


def sobel_horizontal(image_array: np.ndarray) -> np.ndarray:
    """ Return the output of the horizontal Sobel operator with same padding.
        **Note**: This function may return negative values

    Args:
        image_array (np.ndarray): 2D Input array of shape (H, W)

    Returns:
        np.ndarray: Derivative array of shape (H, W).
    """
    img = image_array

    padding = [[0,0],[0,0]]
    kernel = np.asarray([[-1,-2,-1],[0,0,0],[1,2,1]])

    new_img = apply_filter(img, kernel, padding)
    pad_x = (img.shape[0] - new_img.shape[0]) // 2
    pad_y = (img.shape[1] - new_img.shape[1]) // 2

    final_img = np.zeros(img.shape)

    final_img[pad_x:pad_x+new_img.shape[0], pad_y:pad_y+new_img.shape[1]] = new_img

    return final_img
