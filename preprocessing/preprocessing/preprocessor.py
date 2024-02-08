import cv2 as cv
import numpy as np

from data_manager.database_manager import DataSample


class Preprocessor:
    def __init__(self, data_sample, batch_size, transforms: []) -> None:
        self.data = data_sample
        self.batch_size = batch_size
        self.transforms = transforms

    def skeletonization(self, data_sample):
        """
        Perform skeletonization on the images in the data sample.
        """
        for idx, filename in enumerate(data_sample.get_filenames()):
            image = data_sample.data_block[idx]
            gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            _, binary_image = cv.threshold(gray_image, 128, 255, cv.THRESH_BINARY_INV)
            thinned = cv.ximgproc.thinning(
                binary_image, thinningType=cv.ximgproc.THINNING_GUOHALL
            )
            data_sample.data_block[idx] = thinned



    def convert_to_binary_batch(self, data_sample, save_paths=None):
        """
        Convert a batch of images to binary and save them.

        Parameters:
        - image_paths (list): List of paths to input images.
        - save_paths (list or None): List of paths to save the binary images. If None, images won't be saved.
        """
        for i, img in enumerate(data_sample.data_block):
            # Convert to binary
            wd, ht = img.size
            pix = np.array(img.convert("1").getdata(), np.uint8)
            bin_img = 1 - (pix.reshape((ht, wd)) / 255.0)

            # Save the binary image if save_paths provided
            if save_paths:
                plt.imshow(bin_img, cmap="gray")
                plt.savefig(save_paths[i])

            # Update the data sample
            self.data.data_block[i] = bin_img

















def convert_to_binary(self, image_path, save_path="binary.png"):
        """
        Convert the image to binary and save it.

        Parameters:
        - image_path (str): Path to the input image.
        - save_path (str): Path to save the binary image. Default is "binary.png".
        """
        img = im.open(image_path)

        # Convert to binary
        wd, ht = img.size
        pix = np.array(img.convert("1").getdata(), np.uint8)
        bin_img = 1 - (pix.reshape((ht, wd)) / 255.0)

    def correct_skew(self, bin_img, limit=180, delta=10, save_path="skew_corrected.png"):
        """
        Correct skew in the binary image and save the result.

        Parameters:
        - bin_img (numpy.ndarray): Binary image array.
        - limit (int): Maximum angle for skew correction. Default is 180.
        - delta (int): Angle step for skew correction. Default is 10.
        - save_path (str): Path to save the skew-corrected image. Default is "skew_corrected.png".
        """
        angles = np.arange(-limit, limit + delta, delta)
        scores = []

        for angle in angles:
            hist, score = self.find_score(bin_img, angle)
            scores.append(score)

        best_score = max(scores)
        best_angle = angles[scores.index(best_score)]

        # Correct skew
        data = inter.rotate(bin_img, best_angle, reshape=False, order=0)
        img2 = im.fromarray((255 * data).astype("uint8")).convert("RGB")
        img2.save(save_path)

    def find_score(self, arr, angle):
        """
        Find the skew correction score for a given angle.

        Parameters:
        - arr (numpy.ndarray): Image array.
        - angle (float): Angle for skew correction.

        Returns:
        - hist (numpy.ndarray): Histogram of the rotated image.
        - score (float): Skew correction score.
        """
        data = inter.rotate(arr, angle, reshape=False, order=0)
        hist = np.sum(data, axis=1)
        score = np.sum((hist[1:] - hist[:-1]) ** 2)
        return hist, score

# Example usage:
# Create an instance of DataSample with your filenames
data_sample_instance = DataSample(filenames=["test_data/img_test.png"])

# Create an instance of Preprocessor
preprocessor_instance = Preprocessor(data_sample_instance, batch_size=1, transforms=[])

# Convert to binary and save
preprocessor_instance.convert_to_binary("test_data/img_test.png", save_path="binary.png")

# Load the binary image from the DataSample instance
binary_image = data_sample_instance.data_block[0]
