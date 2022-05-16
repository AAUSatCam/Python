#!/usr/bin/env python3
from struct import pack
import numpy as np
import bitarray
import toml
import sys

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

    # Zig-zag the table
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

    # Write zig-zagged quantization table to JFIF
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


def export_jfif(data, size=(1920, 1080), out_name="out.bin"):
    ba = bitarray.bitarray()

    # Load tables (quantization and Huffman) from configuration file
    tables = toml.load('tables.toml')["tables"]

    # Write all relevant JFIF segments
    start_image(ba)
    app0(ba)
    quantization_table(ba, tables["quantization"]["luma"])
    quantization_table(ba, tables["quantization"]["chroma"], luminance=False)
    start_of_frame(ba, size[0], size[1])
    huffman_table(ba, tables["huffman"]["dc_0"]["lengths"], tables["huffman"]["dc_0"]["elements"], 0, 0)
    huffman_table(ba, tables["huffman"]["dc_1"]["lengths"], tables["huffman"]["dc_1"]["elements"], 0, 1)
    huffman_table(ba, tables["huffman"]["ac_0"]["lengths"], tables["huffman"]["ac_0"]["elements"], 1, 0)
    huffman_table(ba, tables["huffman"]["ac_1"]["lengths"], tables["huffman"]["ac_1"]["elements"], 1, 1)
    start_of_scan(ba)

    ba.extend(data)

    end_image(ba)

    with open(out_name, 'wb') as f:
        ba.tofile(f) 

if __name__ == '__main__':
    f_name = sys.argv[1]
    f_name_out = f_name.replace('bin', 'jpg')

    # Load the entropy coded binray data, convert to bitarray and export to JFIF
    ba = bitarray.bitarray()

    with open(f_name, 'rb') as f:
        ba.fromfile(f)

    

    export_jfif(ba, out_name=f_name_out)