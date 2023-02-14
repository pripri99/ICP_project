import numpy as np


def oilpaint(img_numpy, brush_radius, effect_intensity):
    img_height, img_width, _ = img_numpy.shape
    painted_image = np.zeros((img_height, img_width, 3), dtype=np.int32)
    # nRadius pixels are avoided from left, right top, and bottom edges.
    for i in range(brush_radius, img_height - brush_radius):
        for j in range(brush_radius, img_width - brush_radius):
            # Find intensities of nearest nRadius pixels in four direction.
            intensity_count = np.zeros(256, dtype=np.int32)
            sum_r = np.zeros(256, dtype=np.float64)
            sum_g = np.zeros(256, dtype=np.float64)
            sum_b = np.zeros(256, dtype=np.float64)
            for stroke_y in range(-brush_radius, brush_radius + 1):
                for stroke_x in range(-brush_radius, brush_radius + 1):
                    red_val, green_val, blue_val = img_numpy[i + stroke_y][j + stroke_x]

                    # Find intensity of RGB value and apply intensity level.
                    currentIntensity = int(
                        ((red_val + green_val + blue_val) / 3)
                        * effect_intensity
                        / 255
                    )
                    if currentIntensity > 255:
                        currentIntensity = 255
                    intensity_count[currentIntensity] += 1
                    sum_r[currentIntensity] += red_val
                    sum_g[currentIntensity] += green_val
                    sum_b[currentIntensity] += blue_val

            current_max = 0
            max_index = 0
            for index in range(256):
                if intensity_count[index] > current_max:
                    current_max = intensity_count[index]
                    max_index = index

            r = int(sum_r[max_index] / current_max)
            g = int(sum_g[max_index] / current_max)
            b = int(sum_b[max_index] / current_max)
            painted_image[i][j] = np.array([r, g, b])
    return painted_image
