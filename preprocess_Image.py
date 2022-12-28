import cv2
import numpy as np

class preprocess_Image():

    # read image if we want to test an image file
    def read(self, image_path):
        image = cv2.imread(image_path)
        return image

    #  Transform to “LAB” color space
    def BGR2LAB(self, img, type = "LAB"):
        Converted = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        return Converted

    # Increasing contrast
    def increase_contrast(self, img, contrast = 0):
        if contrast != 0:
            f = 131*(contrast + 127)/(127*(131-contrast))
            alpha_c = f
            gamma_c = 127*(1-f)
            buf = img.copy()
            contrasted = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
            return contrasted
    
    def apply_morphology (self, img):

        # apply morphology close
        kernel = np.ones((9,9), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

        # apply morphology open
        kernel = np.ones((9,9), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        return img

    def highlight_target(self, img, contrast= 100, blur_loops = 4, gaussianblur_size = (3,3), medianblur_size = 15):
        # Increasing contrast
        img = self.increase_contrast(img, contrast)

        # Blurring several times to get rid of noise caused by the colored “dots” and borders and gaps between the dots.
        for j in range(blur_loops):
            img = cv2.GaussianBlur(img, gaussianblur_size, cv2.BORDER_DEFAULT)
        img = cv2.medianBlur(img, medianblur_size)

        # thresholding to get binary values
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        img = self.apply_morphology(img)
        return img