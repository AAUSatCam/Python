from cmath import log10, sqrt
from statistics import mean
import glob
from PIL import Image, ImageDraw
import os.path
import numpy as np

img_glob = glob.glob("./Originals/*")
compressed_path = "./Compressed/"

def mae(org_path, compress_path):
    org_image = Image.open(org_path)
    compressed_image = Image.open(compress_path)

    mae_sum = 0

    for x in range(compressed_image.width):
        for y in range(compressed_image.height):
            org_col = org_image.getpixel((x, y))
            compress_col = compressed_image.getpixel((x, y))

            r_diff = abs(org_col[0] - compress_col[0])
            g_diff = abs(org_col[1] - compress_col[1])
            b_diff = abs(org_col[2] - compress_col[2])

            partial_sum = r_diff + g_diff + b_diff

            mae_sum += partial_sum

    
    mae_result = mae_sum / (3 * compressed_image.width * compressed_image.height)



    return mae_result

def mse(org_path, compress_path):
    org_image = Image.open(org_path)
    compressed_image = Image.open(compress_path)

    mse_sum = 0

    for x in range(compressed_image.width):
        for y in range(compressed_image.height):
            org_col = org_image.getpixel((x, y))
            compress_col = compressed_image.getpixel((x, y))

            r_diff = pow(org_col[0] - compress_col[0], 2)
            g_diff = pow(org_col[1] - compress_col[1], 2)
            b_diff = pow(org_col[2] - compress_col[2], 2)

            partial_sum = r_diff + g_diff + b_diff

            mse_sum += partial_sum

    
    mse_result = mse_sum / (3 * compressed_image.width * compressed_image.height)



    return mse_result


def psnr(org_path, compress_path):
    MSE = mse(org_path, compress_path)

    psnr_result = 10*log10(pow(255, 2)/MSE)

    return psnr_result

def mean_gray(img_path):
    image = Image.open(img_path)

    data = np.asarray(image)

    r_gray = np.mean(data[:,:,2])
    g_gray = np.mean(data[:,:,1])
    b_gray = np.mean(data[:,:,0])


    return r_gray, g_gray, b_gray

def corr(org_path, compress_path):
    mean_gray_org = mean_gray(org_path)
    mean_gray_super = mean_gray(compress_path)

    org_image = Image.open(org_path)
    compressed_image = Image.open(compress_path)

    # [RED, GREEN, BLUE]
    a_sums = [0, 0, 0]
    b_sums = [0, 0, 0]
    c_sums = [0, 0, 0]

    for x in range(compressed_image.width):
        for y in range(compressed_image.height):
            a_sums[0] += org_image.getpixel((x, y))[0] * compressed_image.getpixel((x, y))[0]
            a_sums[1] += org_image.getpixel((x, y))[1] * compressed_image.getpixel((x, y))[1]
            a_sums[2] += org_image.getpixel((x, y))[2] * compressed_image.getpixel((x, y))[2]

            b_sums[0] += pow(org_image.getpixel((x, y))[0], 2)
            b_sums[1] += pow(org_image.getpixel((x, y))[1], 2)
            b_sums[2] += pow(org_image.getpixel((x, y))[2], 2)

            c_sums[0] += pow(compressed_image.getpixel((x, y))[0], 2)
            c_sums[1] += pow(compressed_image.getpixel((x, y))[1], 2)
            c_sums[2] += pow(compressed_image.getpixel((x, y))[2], 2)

    pixels = compressed_image.width * compressed_image.height
    
    r_corr = abs((a_sums[0] - pixels * mean_gray_org[0] * mean_gray_super[0]) / (sqrt(b_sums[0] - pixels * mean_gray_org[0]) * sqrt(c_sums[0] - pixels * mean_gray_super[0])))
    g_corr = abs((a_sums[1] - pixels * mean_gray_org[1] * mean_gray_super[1]) / (sqrt(b_sums[1] - pixels * mean_gray_org[1]) * sqrt(c_sums[1] - pixels * mean_gray_super[1])))
    b_corr = abs((a_sums[2] - pixels * mean_gray_org[2] * mean_gray_super[2]) / (sqrt(b_sums[2] - pixels * mean_gray_org[2]) * sqrt(c_sums[2] - pixels * mean_gray_super[2])))

    sum_corr = r_corr + g_corr + b_corr
    avg_corr = sum_corr / 3

    return avg_corr



    
result_file = open(os.path.join(compressed_path, 'superpixel_bench_result.txt'), 'w')
MAE_LIST = []
PSNR_LIST = []
CORR_LIST = []

# For each test-file in ./Originals compare to equivalent superpixel-image
for f in img_glob:
    in_file_basename = os.path.basename(f)
    compressed_image_path = os.path.join(compressed_path, in_file_basename.split('_')[0] + "_JPEG.jpg")
    
    MAE = mae(f, compressed_image_path)
    PSNR = (psnr(f, compressed_image_path)).real
    CORR = corr(f, compressed_image_path)
    result_file.write(f"{f}:{compressed_image_path}, {MAE} (MAE), {PSNR} (PSNR), {CORR} (Correlation)\n")

    MAE_LIST.append(MAE)
    PSNR_LIST.append(PSNR)
    CORR_LIST.append(CORR)

MAE_AVG = mean(MAE_LIST)
PSNR_AVG = mean(PSNR_LIST)
CORR_AVG = mean(CORR_LIST)
    
result_file.write(f"Avg. {MAE_AVG} (MAE), {PSNR_AVG} (PSNR), {CORR_AVG} (Correlation)")
# bayer = bayer / (2**16 - 1)*10

# plt.imshow(data)
# plt.imshow(bayer)
# plt.show() 
 
