from calendar import c
from curses import meta
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import dctn, idctn
import toml
import bitarray
from struct import pack

DEBUG = False

# Calculates how many pixels are missing in a given dimension to be divisble by 8
def missing_reps(dim):
    dim_modulo = dim % 8

    if dim_modulo != 0:
        return (8 - dim_modulo)
    else:
        return 0

# Extends image either horizontally or vertically with the last column or row of pixels respectively. Horizontal if hor=True, else vertical.
def extend_image(img, reps, hor=True):
    if hor:
        last_column = np.array([img[:, -1, :]]).reshape((img.shape[0], 1, 3))
        last = np.repeat(last_column, repeats=reps, axis=1)
        return np.hstack([img, last])
    else:
        last_row = np.array([img[-1, :, :]])
        last = np.repeat(last_row, repeats=reps, axis=0)
        return np.vstack([img, last])


def image_iter(img):
    # Dimensions of the original image
    image_height, image_width = img.shape[:2]
    # Dimensions of a single block
    block_height, block_width = 8, 8

    # How many columns to extend by, h_rep. How many rows to extend by, v_rep.
    h_rep = missing_reps(image_width)
    v_rep = missing_reps(image_height)

    # Extend the image, so the dimensions are divisible by 8
    img_v_extend = extend_image(img, v_rep, hor=False)
    img_h_extend = extend_image(img_v_extend, h_rep)


    # Update image-dimensions to reflect the extended image
    image_height, image_width = img_h_extend.shape[:2]

    # Yield meta-blocks (8, 8, 3) blocks
    for row in np.arange(image_height - block_height + 1, step=block_height):
        for col in np.arange(image_width - block_width + 1, step=block_width):
            yield img_h_extend[row:row+block_height, col:col+block_width]


# Level shifts matrix by 128 before DCT
def level_shift_forward(meta_block):
    #print("Level_shift_backward", meta_block.dtype)
    return np.int8(meta_block - 128)


def level_shift_backward(meta_block):
    # Convert to unsigned 16 bit int to avoid overflow. Clip the values to uint8 values and then convert to uint8.
    meta_block = meta_block.astype(np.uint16)
    return np.clip(meta_block + 128, a_min=0, a_max=255).astype(np.uint8)


def dct_forward(meta_block):
    #print("dct_forward", meta_block.dtype)
    dct = dctn(meta_block, norm='ortho', axes=[0, 1])
    rounded_dct = np.rint(dct)

    return rounded_dct


def dct_backward(meta_block):
    #print("dct_backward", meta_block.dtype)
    dct_back = idctn(meta_block, norm='ortho')
    rounded_dct = np.rint(dct_back)

    return rounded_dct


# Quantize DCT-coefficients in meta_block (8, 8, 3) given dictionary of two tables 'luma' and 'chroma'
def quantize(meta_block, tables):
    #print("quantize", meta_block.dtype)
    meta_block[:, :, 0] = meta_block[:, :, 0] / (np.array(tables["luma"]))
    meta_block[:, :, 1] = meta_block[:, :, 1] / (np.array(tables["chroma"]))
    meta_block[:, :, 2] = meta_block[:, :, 2] / (np.array(tables["chroma"]))

    return np.int16(np.rint(meta_block))


def dequantize(meta_block, tables):
    #print("dequantize", meta_block.dtype)
    meta_block[:, :, 0] = meta_block[:, :, 0] * (np.array(tables["luma"]))
    meta_block[:, :, 1] = meta_block[:, :, 1] * (np.array(tables["chroma"]))
    meta_block[:, :, 2] = meta_block[:, :, 2] * (np.array(tables["chroma"]))

    return meta_block

def transform_iter(block_iterator):
    tables = toml.load('tables.toml')["tables"]
    for block in block_iterator:
        block_shifted = level_shift_forward(block)
        block_transformed = dct_forward(block_shifted)
        block_quantized = quantize(block_transformed, tables['quantization'])

        yield block_quantized


def zigzag(meta_block):
    a = meta_block[:, :, 0].flatten()
    b = meta_block[:, :, 1].flatten()
    c = meta_block[:, :, 2].flatten()

    ZIGZAGINVERSE = np.array([[0,  1,  5,  6,  14, 15, 27, 28],
                   [2,  4,  7,  13, 16, 26, 29, 42],
                   [3,  8,  12, 17, 25, 30, 41, 43],
                   [9,  11, 18, 24, 31, 40, 44,53],
                   [10, 19, 23, 32, 39, 45, 52,54],
                   [20, 22, 33, 38, 46, 51, 55,60],
                   [21, 34, 37, 47, 50, 56, 59,61],
                   [35, 36, 48, 49, 57, 58, 62,63]])

    
    ZIGZAGFLATINVERSE = ZIGZAGINVERSE.flatten()
    ZIGZAGFLAT = np.argsort(ZIGZAGFLATINVERSE)

    return np.array([a[ZIGZAGFLAT], b[ZIGZAGFLAT], c[ZIGZAGFLAT]])


# Coding, huffman tables
dc_luma_table = ["00", "010", "011", "100", "101", "110", "1110", "11110", "111110", "1111110", "11111110", "111111110"]
dc_chroma_table = ["00", "01", "10", "110", "1110", "11110", "111110","1111110", "11111110", "111111110", "1111111110", "11111111110"]

# def encode_iter(block_iterator):
#     ba = bitarray.bitarray()
#     previous_dc = None
#     dpcm = []
#     for block in block_iterator:
#         zigzag_block = zigzag(block)

#         luma = zigzag_block[0]
#         chroma_cb = zigzag_block[1]
#         chroma_cr = zigzag_block[2]

#         luma_dc = luma[0]
#         chroma_cb_dc = chroma_cb[0]
#         chroma_cr_dc = chroma_cr[0]

#         if not previous_dc:
#             luma_bits = np.ceil(np.log2(np.abs(luma_dc) + 1)).astype(np.uint8)
#             cb_bits = np.ceil(np.log2(np.abs(chroma_cb_dc) + 1)).astype(np.uint8)
#             cr_bits = np.ceil(np.log2(np.abs(chroma_cr_dc) + 1)).astype(np.uint8)

#             # If the first block, don't calculate difference, encode the amplitude instead
            
#             ba.extend(dc_luma_table[luma_bits])
#             if luma_dc == 0:
#                 # If dc-component is 0, don't append anything to the huffman code
#                 pass
#             elif luma_dc > 0:
#                 # If dc-component is positive, append ´bits´ LSB of luma_dc
#                 ba.extend(bin(luma_dc)[2:])
#             else:
#                 # If dc-component is negative, ´bits´ LSB of append luma_dc - 1 two's complement
#                 bin_repr = bin(-luma_dc)[2:].replace('1', '2').replace('0', '1').replace('2', '0')
#                 ba.extend(bin_repr)
#             ba.extend("1010")
            
#             ba.extend(dc_chroma_table[cb_bits])
#             if chroma_cb_dc == 0:
#                 # If dc-component is 0, don't append anything to the huffman code
#                 pass
#             elif chroma_cb_dc > 0:
#                 # If dc-component is positive, append ´bits´ LSB of luma_dc
#                 ba.extend(bin(chroma_cb_dc)[2:])
#             else:
#                 # If dc-component is negative, ´bits´ LSB of append luma_dc - 1 two's complement
#                 bin_repr = bin(-chroma_cb_dc)[2:].replace('1', '2').replace('0', '1').replace('2', '0')
#                 ba.extend(bin_repr)
#             ba.extend("00")
            
#             ba.extend(dc_chroma_table[cr_bits])
#             if chroma_cr_dc == 0:
#                 # If dc-component is 0, don't append anything to the huffman code
#                 pass
#             elif chroma_cr_dc > 0:
#                 # If dc-component is positive, append ´bits´ LSB of luma_dc
#                 ba.extend(bin(chroma_cr_dc)[2:])
#             else:
#                 # If dc-component is negative, ´bits´ LSB of append luma_dc - 1 two's complement
#                 bin_repr = bin(-chroma_cr_dc)[2:].replace('1', '2').replace('0', '1').replace('2', '0')
#                 ba.extend(bin_repr)
#             ba.extend("00")
            


#         else:
#             dpcm_diff_luma = np.int16(luma_dc) - np.int16(previous_dc[0])
#             dpcm_diff_cb = np.int16(chroma_cb_dc) - np.int16(previous_dc[1])
#             dpcm_diff_cr = np.int16(chroma_cr_dc) - np.int16(previous_dc[2])


#             luma_bits = np.ceil(np.log2(np.abs(dpcm_diff_luma) + 1)).astype(np.uint8)
#             cb_bits = np.ceil(np.log2(np.abs(dpcm_diff_cb) + 1)).astype(np.uint8)
#             cr_bits = np.ceil(np.log2(np.abs(dpcm_diff_cr) + 1)).astype(np.uint8)

#             # If the first block, don't calculate difference, encode the amplitude instead
            
#             ba.extend(dc_luma_table[luma_bits])
#             if dpcm_diff_luma == 0:
#                 # If dc-component is 0, don't append anything to the huffman code
#                 pass
#             elif dpcm_diff_luma > 0:
#                 # If dc-component is positive, append ´bits´ LSB of luma_dc
#                 ba.extend(bin(dpcm_diff_luma)[2:])
#             else:
#                 # If dc-component is negative, ´bits´ LSB of append luma_dc - 1 two's complement
#                 bin_repr = bin(-dpcm_diff_luma)[2:].replace('1', '2').replace('0', '1').replace('2', '0')
#                 ba.extend(bin_repr)
#             ba.extend("1010")
            
#             ba.extend(dc_chroma_table[cb_bits])
#             if dpcm_diff_cb == 0:
#                 # If dc-component is 0, don't append anything to the huffman code
#                 pass
#             elif dpcm_diff_cb > 0:
#                 # If dc-component is positive, append ´bits´ LSB of luma_dc
#                 ba.extend(bin(dpcm_diff_cb)[2:])
#             else:
#                 # If dc-component is negative, ´bits´ LSB of append luma_dc - 1 two's complement
#                 bin_repr = bin(-dpcm_diff_cb)[2:].replace('1', '2').replace('0', '1').replace('2', '0')
#                 ba.extend(bin_repr)
#             ba.extend("00")
            
#             ba.extend(dc_chroma_table[cr_bits])
#             if dpcm_diff_cr == 0:
#                 # If dc-component is 0, don't append anything to the huffman code
#                 pass
#             elif dpcm_diff_cr > 0:
#                 # If dc-component is positive, append ´bits´ LSB of luma_dc
#                 ba.extend(bin(dpcm_diff_cr)[2:])
#             else:
#                 # If dc-component is negative, ´bits´ LSB of append luma_dc - 1 two's complement
#                 bin_repr = bin(-dpcm_diff_cr)[2:].replace('1', '2').replace('0', '1').replace('2', '0')
#                 ba.extend(bin_repr)
#             ba.extend("00")
        
#         previous_dc = (luma_dc, chroma_cb_dc, chroma_cr_dc)
    
#     return ba

# NEW IMPL
dc_luma_table = ["00", "010", "011", "100", "101", "110", "1110", "11110", "111110", "1111110", "11111110", "111111110"]
dc_chroma_table = ["00", "01", "10", "110", "1110", "11110", "111110","1111110", "11111110", "111111110", "1111111110", "11111111110"]

ac_luma_table = ["1010", "00", "01", "100", "1011", "11010", "1111000", "11111000", "1111110110", "1111111110000010", "1111111110000011", "1100", "11011", "1111001", "111110110", "11111110110", "1111111110000100", "1111111110000101", "1111111110000110", "1111111110000111", "1111111110001000", "11100", "11111001", "1111110111", "111111110100", "1111111110001001", "1111111110001010", "1111111110001011", "1111111110001100", "1111111110001101", "1111111110001110", "111010", "111110111", "111111110101", "1111111110001111", "1111111110010000", "1111111110010001", "1111111110010010", "1111111110010011", "1111111110010100", "1111111110010101", "111011", "1111111000", "1111111110010110", "1111111110010111", "1111111110011000", "1111111110011001", "1111111110011010", "1111111110011011", "1111111110011100", "1111111110011101", "1111010", "11111110111", "1111111110011110", "1111111110011111", "1111111110100000", "1111111110100001", "1111111110100010", "1111111110100011", "1111111110100100", "1111111110100101", "1111011", "111111110110", "1111111110100110", "1111111110100111", "1111111110101000", "1111111110101001", "1111111110101010", "1111111110101011", "1111111110101100", "1111111110101101", "11111010", "111111110111", "1111111110101110", "1111111110101111", "1111111110110000", "1111111110110001", "1111111110110010", "1111111110110011", "1111111110110100", "1111111110110101", "111111000", "111111111000000", "1111111110110110", "1111111110110111", "1111111110111000", "1111111110111001", "1111111110111010", "1111111110111011", "1111111110111100", "1111111110111101", "111111001", "1111111110111110", "1111111110111111", "1111111111000000", "1111111111000001", "1111111111000010", "1111111111000011", "1111111111000100", "1111111111000101", "1111111111000110", "111111010", "1111111111000111", "1111111111001000", "1111111111001001", "1111111111001010", "1111111111001011", "1111111111001100", "1111111111001101", "1111111111001110", "1111111111001111", "1111111001", "1111111111010000", "1111111111010001", "1111111111010010", "1111111111010011", "1111111111010100", "1111111111010101", "1111111111010110", "1111111111010111", "1111111111011000", "1111111010", "1111111111011001", "1111111111011010", "1111111111011011", "1111111111011100", "1111111111011101", "1111111111011110", "1111111111011111", "1111111111100000", "1111111111100001", "11111111000", "1111111111100010", "1111111111100011", "1111111111100100", "1111111111100101", "1111111111100110", "1111111111100111", "1111111111101000", "1111111111101001", "1111111111101010", "1111111111101011", "1111111111101100", "1111111111101101", "1111111111101110", "1111111111101111", "1111111111110000", "1111111111110001", "1111111111110010", "1111111111110011", "1111111111110100", "1111111111110101", "1111111111110110", "1111111111110111", "1111111111111000", "1111111111111001", "1111111111111010", "1111111111111011", "1111111111111100", "1111111111111101", "1111111111111110", "11111111001"]
ac_chroma_table = ["00", "01", "100", "1010", "11000", "11001", "111000", "1111000", "111110100", "1111110110", "111111110100", "1011", "111001", "11110110", "111110101", "11111110110", "111111110101", "1111111110001000", "1111111110001001", "1111111110001010", "1111111110001011", "11010", "11110111", "1111110111", "111111110110", "111111111000010", "1111111110001100", "1111111110001101", "1111111110001110", "1111111110001111", "1111111110010000", "11011", "11111000", "1111111000", "111111110111", "1111111110010001", "1111111110010010", "1111111110010011", "1111111110010100", "1111111110010101", "1111111110010110", "111010", "111110110", "1111111110010111", "1111111110011000", "1111111110011001", "1111111110011010", "1111111110011011", "1111111110011100", "1111111110011101", "1111111110011110", "111011", "1111111001", "1111111110011111", "1111111110100000", "1111111110100001", "1111111110100010", "1111111110100011", "1111111110100100", "1111111110100101", "1111111110100110", "1111001", "11111110111", "1111111110100111", "1111111110101000", "1111111110101001", "1111111110101010", "1111111110101011", "1111111110101100", "1111111110101101", "1111111110101110", "1111010", "11111111000", "1111111110101111", "1111111110110000", "1111111110110001", "1111111110110010", "1111111110110011", "1111111110110100", "1111111110110101", "1111111110110110", "11111001", "1111111110110111", "1111111110111000", "1111111110111001", "1111111110111010", "1111111110111011", "1111111110111100", "1111111110111101", "1111111110111110", "1111111110111111", "111110111", "1111111111000000", "1111111111000001", "1111111111000010", "1111111111000011", "1111111111000100", "1111111111000101", "1111111111000110", "1111111111000111", "1111111111001000", "111111000", "1111111111001001", "1111111111001010", "1111111111001011", "1111111111001100", "1111111111001101", "1111111111001110", "1111111111001111", "1111111111010000", "1111111111010001", "111111001", "1111111111010010", "1111111111010011", "1111111111010100", "1111111111010101", "1111111111010110", "1111111111010111", "1111111111011000", "1111111111011001", "1111111111011010", "111111010", "1111111111011011", "1111111111011100", "1111111111011101", "1111111111011110", "1111111111011111", "1111111111100000", "1111111111100001", "1111111111100010", "1111111111100011", "11111111001", "1111111111100100", "1111111111100101", "1111111111100110", "1111111111100111", "1111111111101000", "1111111111101001", "1111111111101010", "1111111111101011", "1111111111101100", "11111111100000", "1111111111101101", "1111111111101110", "1111111111101111", "1111111111110000", "1111111111110001", "1111111111110010", "1111111111110011", "1111111111110100", "1111111111110101", "111111111000011", "1111111111110110", "1111111111110111", "1111111111111000", "1111111111111001", "1111111111111010", "1111111111111011", "1111111111111100", "1111111111111101", "1111111111111110", "1111111010"]
def encode_amplitude(amp):
    if amp == 0:
        # If the amp is 0, nothing is appended after the category
        bin_repr = ""
    elif amp > 0:
        # If the diff is positive, append 'dc_category' amount of LSB of diff
        bin_repr = bin(amp)[2:]
    else:
        # If the diff is negative, append 'dc_category' amount of LSB of (diff - 1) in two's complement (corresponding to one's complement)
        bin_repr = bin(-amp)[2:].replace('1', '2').replace('0', '1').replace('2', '0')
    
    return bin_repr

def huffman_encode_dc(bit_array, dc_diffs, channel):
    # Expects dc_values to be (3,)-dimensional array of the three channels' DC-diff values and channel from 0-2

        diff = dc_diffs[channel]
        dc_category = int(diff).bit_length()

        # If DC value is from luma channel, the Huffman code is determined differently from the other channels
        if channel == 0:
            base_code = dc_luma_table[dc_category]
        else:
            base_code = dc_chroma_table[dc_category]
        
        # The base code is added to the bitarray
        bit_array.extend(base_code)

        # Appending the amplitude after the category
        amp_encoding = encode_amplitude(diff)
        bit_array.extend(amp_encoding)
        if DEBUG:
            print(f"diff: {diff}, amp_encoding: {amp_encoding}, dc_cat: {dc_category}, base_code: {base_code}, chan: {channel}")
        
        if diff == 0:
            return True
        else:
            return False


def huffman_encode_ac(bit_array, ac_values, channel):
    # Expects dc_values to be (3, 63)-dimensional array of the three channels' AC values

    zero_count = 0
    zrl_count = 0

    for value in ac_values[channel]:
        # If 0 is encountered, add to zero run-length, otherwise encode amplitude and run_length
        if value == 0:
            zero_count += 1
        else:
            # A non-zero value was encountered, which has to be encoded. The run length is equal to the zero_count
            while zrl_count > 0:
                zrl_count -= 1
                if channel == 0:
                    # If luma
                    zrl_code = ac_luma_table[-1]
                else:
                    # If chroma
                    zrl_code = ac_chroma_table[-1]
                if DEBUG:
                    print("ZRL")
                bit_array.extend(zrl_code)

            ac_category = int(value).bit_length()
            table_index = zero_count * 10 + ac_category

            if channel == 0:
                # If luma
                base_code = ac_luma_table[table_index]
            else:
                # If chroma
                base_code = ac_chroma_table[table_index]
            
            bit_array.extend(base_code)

            # Encode the amplitude
            amp_encoding = encode_amplitude(value)
            bit_array.extend(amp_encoding)
            if DEBUG:
                print(f"Value: {value}, amp_encoding: {amp_encoding}, zero_count: {zero_count}, ac_cat: {ac_category}, tab_index: {table_index}, code: {base_code}, chan: {channel}")

            zero_count = 0
            continue

        # If 16 zeros have been encountered in a row, reset zero counter and add ZRL
        if zero_count == 16:
            zero_count = 0
            zrl_count += 1

    # End of block reached, for-loop exited. Is zero_count different from 0, then add EOB
    if DEBUG:
        print("zero_count", zero_count)
    if zero_count != 0 or zrl_count != 0:
        if channel == 0:
            # If luma
            eob_code = ac_luma_table[0]
        else:
            # If chroma
            eob_code = ac_chroma_table[0]
        if DEBUG:
            print("EOB")
        bit_array.extend(eob_code)


def stuff_data(ba):
    ba_bytes = ba.tobytes()
    out_bytes = bytearray()

    for byte in ba_bytes:
        out_bytes.append(byte)

        if byte == 0xFF:
            out_bytes.append(0x00)
    
    
    result_ba = bitarray.bitarray()
    result_ba.frombytes(bytes(out_bytes))

    return result_ba

def encode_iter(block_iterator):
    global DEBUG
    ba = bitarray.bitarray()
    previous_dc_values = np.array([])

    for block in block_iterator:
        if len(ba) > 192900:
            DEBUG = True
        if len(ba) > 193300:
            DEBUG = False
        # print(block[:, :, 0])
        # print(block[:, :, 1])
        # print(block[:, :, 2])
        # break
        zigzag_block = zigzag(block)

        # Extract three DC-values of the current meta-block as a (3,) np array.
        dc_values = zigzag_block[:, 0]

        # Extract the remaining 63 AC components.
        ac_values = zigzag_block[:, 1:]

        for i in range(3):
            # If this is the first block in the image, diff is not calculated.            
            if not previous_dc_values.any():
                dc_values_enc = dc_values
            else:
                dc_values_enc = dc_values - previous_dc_values

            huffman_encode_dc(ba, dc_values_enc, i)
            huffman_encode_ac(ba, ac_values, i)

        previous_dc_values = dc_values

        # if DEBUG:
        #     break

    # All blocks have been encoded. Pad with 1 to align next marker.
    bit_count = len(ba) % 8

    if bit_count == 0:
        # The next marker is already aligned.
        pass
    else:
        pad_length = 8 - bit_count
        ba.extend('1' * pad_length)
    
    return ba

# ALL JFIF-RELATED FUNCTIONS
def start_image(bit_array):
    bit_array.frombytes(pack(">H", 0xFFD8))

def end_image(bit_array):
    bit_array.frombytes(pack(">H", 0xFFD9))

def app0(bit_array):
    # FFE0 APP0 marker
    bit_array.frombytes(pack(">H", 0xFFE0))
    
    # Length of the segment, 16
    bit_array.frombytes(pack(">H", 16))

    # Identifier, JFIF
    bit_array.frombytes(b"JFIF\0")

    # Version, 1.1
    bit_array.frombytes(pack(">H", 0x0101))

    # Units, 1 (dpi)
    bit_array.frombytes(pack("B", 1))

    # Density, 72x72
    bit_array.frombytes(pack(">H", 0x0048))
    bit_array.frombytes(pack(">H", 0x0048))

    # Thumbnail
    bit_array.frombytes(pack(">H", 0x0000))

def quantization_table(bit_array, table, luminance=True):
    # FFDB Quantization table marker
    bit_array.frombytes(pack(">H",0xFFDB))

    # Length of the segment, 67
    bit_array.frombytes(pack(">H", 67))

    # If luminance the id is 0, if chrominance the id is 1
    if luminance:
        bit_array.frombytes(pack("B", 0))
    else:
        bit_array.frombytes(pack("B", 1))

    ZIGZAGINVERSE = np.array([[0,  1,  5,  6,  14, 15, 27, 28],
                   [2,  4,  7,  13, 16, 26, 29, 42],
                   [3,  8,  12, 17, 25, 30, 41, 43],
                   [9,  11, 18, 24, 31, 40, 44,53],
                   [10, 19, 23, 32, 39, 45, 52,54],
                   [20, 22, 33, 38, 46, 51, 55,60],
                   [21, 34, 37, 47, 50, 56, 59,61],
                   [35, 36, 48, 49, 57, 58, 62,63]])

    
    ZIGZAGFLATINVERSE = ZIGZAGINVERSE.flatten()
    ZIGZAGFLAT = np.argsort(ZIGZAGFLATINVERSE)

    values = np.array(table).flatten()[ZIGZAGFLAT]

    bit_array.frombytes(pack(">64B", *values))

def huffman_table(bit_array, lengths, values, type_class, destination):
    # FFC4 Huffman table marker
    bit_array.frombytes(pack(">H",0xFFC4))

    # Length of the segment, len(huffval) + 19
    bit_array.frombytes(pack(">H", len(values) + 19))

    # Class (DC = 0, AC = 1) and Destination (Luma = 0, Chroma = 1)
    bit_array.frombytes(pack("B", 16 * type_class + destination))

    # BITS, how many huffman codes of i=1...16 length
    for l in lengths:
        bit_array.frombytes(pack("B", l))

    # HUFFVALUES, the symbols each code corresponds to
    for v in values:
        bit_array.frombytes(pack("B", int(v, 16)))

def start_of_frame(bit_array, width, height):
    # FFC0 Start of Frame baseline marker
    bit_array.frombytes(pack(">H", 0xFFC0))

    # Length of the segment, 17
    bit_array.frombytes(pack(">H", 17))

    # Color depth, 8 bit
    bit_array.frombytes(pack("B", 8))

    # Height of the image, 'height'
    bit_array.frombytes(pack(">H", height))

    # Width of the image, 'width,
    bit_array.frombytes(pack(">H", width))

    # Amount of components, 3
    bit_array.frombytes(pack("B", 3))

    components = [
        [(1, 1), 0],
        [(1, 1), 1],
        [(1, 1), 1]
    ]

    for i, c in enumerate(components):
        hi_lo = c[0]
        bit_array.frombytes(pack("B", i + 1))
        bit_array.frombytes(pack("B", 16 * hi_lo[0] + hi_lo[1]))
        bit_array.frombytes(pack("B", c[1]))

def start_of_scan(bit_array):
    # FFDA Start of Scan marker
    bit_array.frombytes(pack(">H", 0xFFDA))

    # Length of the segment, 12
    bit_array.frombytes(pack(">H", 12))

    # Amount of components, 3
    bit_array.frombytes(pack("B", 3))

    # Destination selector (DC, AC)
    components = [
        (0, 0),
        (1, 1),
        (1, 1)
    ]

    for i, c in enumerate(components):
        bit_array.frombytes(pack("B", i + 1))
        bit_array.frombytes(pack("B", 16 * c[0] + c[1]))


    # Spectral selection (not entirely sure what this is)
    bit_array.frombytes(pack(">H", 0x003F))
    bit_array.frombytes(pack("B", 0))


def export_jfif(data):
    ba = bitarray.bitarray()

    # Load tables (quantization and Huffman) from configuration file
    tables = toml.load('tables.toml')["tables"]
    start_image(ba)
    app0(ba)
    quantization_table(ba, tables["quantization"]["luma"])
    quantization_table(ba, tables["quantization"]["chroma"], luminance=False)
    start_of_frame(ba, 1920, 1080)
    huffman_table(ba, tables["huffman"]["dc_0"]["lengths"], tables["huffman"]["dc_0"]["elements"], 0, 0)
    huffman_table(ba, tables["huffman"]["dc_1"]["lengths"], tables["huffman"]["dc_1"]["elements"], 0, 1)
    huffman_table(ba, tables["huffman"]["ac_0"]["lengths"], tables["huffman"]["ac_0"]["elements"], 1, 0)
    huffman_table(ba, tables["huffman"]["ac_1"]["lengths"], tables["huffman"]["ac_1"]["elements"], 1, 1)
    start_of_scan(ba)

    print(data[:100])
    ba.extend(data)

    end_image(ba)

    with open("out_image.bin", 'wb') as f:
        ba.tofile(f)

if __name__ == '__main__':
    # Load an uncompressed image, convert it to YCbrCr, and convert to 3D np array
    unc_img = Image.open('memdump_comp_buf_after.PNG').convert('RGB')
    ycbcr = unc_img.convert('YCbCr')

    data = np.asarray(ycbcr)

    # #Create an iterator over the np array
    block_stream = image_iter(data)
    block_stream_transformed = transform_iter(block_stream)
    entropy_coded = encode_iter(block_stream_transformed)
    # next(block_stream_transformed)
    # first_transformed_block = next(block_stream_transformed)
    # print(first_transformed_block[:, :, 0])
    # print(first_transformed_block[:, :, 1])
    # print(first_transformed_block[:, :, 2])

    # tables = toml.load('tables.toml')["tables"]

    # print(f"Shape: {data.shape[1] // 8}, {188}")
    # with open('plain_decoded_nature_broken.txt', 'w') as f:
    #     for y in range(188):
    #         for x in range(data.shape[1] // 8):
    #             f.write(f"Block: {x}, {y}\n")
    #             a_block = next(block_stream_transformed)
    #             for i in range(3):
    #                 try:
    #                     block = a_block[:, :, i].flatten()
    #                 except:
    #                     break
    #                 line = ", ".join([str(x) for x in block])
    #                 f.write(f'{line}\n')

    #             f.write('\n')
    # reference_im = np.fromfile('nature_matrix_2.bin', dtype=np.int16).reshape((1504, 2000, 3))
    # print(reference_im[:8, 8:16, 0])
    # print(reference_im[:8, 8:16, 1])
    # print(reference_im[:8, 8:16, 2])

    # block_stream = image_iter(reference_im)

    # for i in range(6506):
    #     next(block_stream)

    # for i in range(4):
    #     block = next(block_stream)
    #     zigzag_block = zigzag(block)

    #     print(zigzag_block[0, :])
    #     print(zigzag_block[1, :])
    #     print(zigzag_block[2, :])
    # quit()
    # entropy_coded = encode_iter(block_stream)

    entropy_coded_stuffed = stuff_data(entropy_coded)
    
    export_jfif(entropy_coded_stuffed)  
    # im = Image.open("memdump_comp_buf_after.PNG")
    # im_ycbcr = im.convert('YCbCr')
    
    # block = np.asarray(im)[:8, :8]
    # block_ycbcr = np.asarray(im_ycbcr)[:8, :8] 

    # np.savetxt("block0.txt", block[:, :, 0].astype(np.int16) - 128, newline=";\n", delimiter=", ", fmt="%d")
    # np.savetxt("block1.txt", block[:, :, 1].astype(np.int16) - 128, newline=";\n", delimiter=", ", fmt="%d")
    # np.savetxt("block2.txt", block[:, :, 2].astype(np.int16) - 128, newline=";\n", delimiter=", ", fmt="%d")

    # print("### RAW DATA ###")
    # print(block[:, :, 0])
    # print(block[:, :, 1])
    # print(block[:, :, 2])

    # print("### YCbCr DATA ###")
    # print(block_ycbcr[:, :, 0])
    # print(block_ycbcr[:, :, 1])
    # print(block_ycbcr[:, :, 2])

    # print("### Level Shifted DATA ###")
    # print(block_ycbcr[:, :, 0].astype(np.int16) - 128)
    # print(block_ycbcr[:, :, 1].astype(np.int16) - 128)
    # print(block_ycbcr[:, :, 2].astype(np.int16) - 128)

    # block_ycbcr = block_ycbcr.astype(np.int16) - 128

    # dct_block = dctn(block_ycbcr[:, :], norm='ortho', axes=[0, 1])

    # np.set_printoptions(suppress=True)

    # # print("### DCT DATA ###")
    # # print(dct_block[:, :, 0].astype(np.int16))
    # # print(dct_block[:, :, 1].astype(np.int16))
    # # print(dct_block[:, :, 2].astype(np.int16))

    # quant_block = quantize(dct_block, tables["quantization"])

    # print("### QUANTIZED DATA ###")
    # print(quant_block[:, :, 0].astype(np.int16))
    # print(quant_block[:, :, 1].astype(np.int16))
    # print(quant_block[:, :, 2].astype(np.int16))

    # ### BLOCK 2
    # block_2 = np.asarray(im)[:8, 8:16]
    # block_ycbcr_2 = np.asarray(im_ycbcr)[:8, 8:16] 

    # block_ycbcr_2 = block_ycbcr_2.astype(np.int16) - 128

    # dct_block_2 = dctn(block_ycbcr_2[:, :], norm='ortho', axes=[0, 1])

    # quant_block_2 = quantize(dct_block_2, tables["quantization"])

    # print("### QUANTIZED DATA ###")
    # print(quant_block_2[:, :, 0].astype(np.int16))
    # print(quant_block_2[:, :, 1].astype(np.int16))
    # print(quant_block_2[:, :, 2].astype(np.int16))

    # ### BLOCK 2
    # block_3 = np.asarray(im)[:8, 16:24]
    # block_ycbcr_3 = np.asarray(im_ycbcr)[:8, 16:24] 

    # block_ycbcr_3 = block_ycbcr_3.astype(np.int16) - 128

    # dct_block_3 = dctn(block_ycbcr_3[:, :], norm='ortho', axes=[0, 1])

    # quant_block_3 = quantize(dct_block_3, tables["quantization"])

    # print("### QUANTIZED DATA ###")
    # print(quant_block_3[:, :, 0].astype(np.int16))
    # print(quant_block_3[:, :, 1].astype(np.int16))
    # print(quant_block_3[:, :, 2].astype(np.int16))

    # dct_block = dctn(block[:, :].astype(np.int16) - 128, norm='ortho', axes=[0, 1])
    # print("DCT", dct_block)
    # quantized_block = quantize(dct_block, tables['quantization'])
    # print("QUANT", quantized_block)
    # zigzagged_block = zigzag(quantized_block)
    # np.savetxt('zigzagged_blocks0.txt', zigzagged_block[0], newline=" ", delimiter=" ", fmt="%d")
    # np.savetxt('zigzagged_blocks1.txt', zigzagged_block[1], newline=" ", delimiter=" ", fmt="%d")
    # np.savetxt('zigzagged_blocks2.txt', zigzagged_block[2], newline=" ", delimiter=" ", fmt="%d")

    # print(dct_block[:, :, 0])
    # print(dct_block[:, :, 1])
    # print(dct_block[:, :, 2])

    # for row in np.arange(1504 - 8 + 1, step=8):
    #     for col in np.arange(2000 - 8 + 1, step=8):
    #         block = block_stream[row:row+8, col:col+8]
    #         first_block_shifted = level_shift_forward(block)
    #         first_block_transformed = dct_forward(first_block_shifted)
    #         first_block_quantized = quantize(first_block_transformed, tables['quantization'])

    #         first_block_dequantized = dequantize(first_block_quantized, tables['quantization'])
    #         first_block_intransformed = dct_backward(first_block_dequantized)
    #         first_block_unshifted = level_shift_backward(first_block_intransformed)
    #         block_stream[row:row+8, col:col+8] = first_block_unshifted

    # img = Image.fromarray(block_stream, mode="YCbCr").convert("RGB")
    # img.save("out.png")
    # plt.imshow(img)
    # plt.show()


    # sample_array = np.array([
    #     [154, 123, 123, 123, 123, 123, 123, 136],
    #     [192, 180, 136, 154, 154, 154, 136, 110],
    #     [254, 198, 154, 154, 180, 154, 123, 123],
    #     [239, 180, 136, 180, 180, 166, 123, 123],
    #     [180, 154, 136, 167, 166, 149, 136, 136],
    #     [128, 136, 123, 136, 154, 180, 198, 154],
    #     [123, 105, 110, 149, 136, 136, 180, 166],
    #     [110, 136, 123, 123, 123, 136, 154, 136]
    # ])

#     level_shifted = level_shift_forward(sample_array)
#     f_dct = dct_forward(level_shifted)
#     b_dct = dct_backward(f_dct)
#     level_shifted_b = level_shift_backward(b_dct)

#     print(sample_array)
#     print(level_shifted)
#     print(f_dct)
#     print(b_dct)
#     print(level_shifted_b)
