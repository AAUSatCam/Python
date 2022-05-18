import bitarray
from bitarray.util import ba2int
import numpy as np
import math

X_BLOCK_MIN = 0
X_BLOCK_MAX = 1
Y_BLOCK_MIN = 0
Y_BLOCK_MAX = 1

DEBUG = False

# Huffman tables
dc_luma_table = [bitarray.bitarray(x) for x in ["00", "010", "011", "100", "101", "110", "1110", "11110", "111110", "1111110", "11111110", "111111110"]]
dc_chroma_table = [bitarray.bitarray(x) for x in ["00", "01", "10", "110", "1110", "11110", "111110","1111110", "11111110", "111111110", "1111111110", "11111111110"]]

ac_luma_table = [bitarray.bitarray(x) for x in ["1010", "00", "01", "100", "1011", "11010", "1111000", "11111000", "1111110110", "1111111110000010", "1111111110000011", "1100", "11011", "1111001", "111110110", "11111110110", "1111111110000100", "1111111110000101", "1111111110000110", "1111111110000111", "1111111110001000", "11100", "11111001", "1111110111", "111111110100", "1111111110001001", "1111111110001010", "1111111110001011", "1111111110001100", "1111111110001101", "1111111110001110", "111010", "111110111", "111111110101", "1111111110001111", "1111111110010000", "1111111110010001", "1111111110010010", "1111111110010011", "1111111110010100", "1111111110010101", "111011", "1111111000", "1111111110010110", "1111111110010111", "1111111110011000", "1111111110011001", "1111111110011010", "1111111110011011", "1111111110011100", "1111111110011101", "1111010", "11111110111", "1111111110011110", "1111111110011111", "1111111110100000", "1111111110100001", "1111111110100010", "1111111110100011", "1111111110100100", "1111111110100101", "1111011", "111111110110", "1111111110100110", "1111111110100111", "1111111110101000", "1111111110101001", "1111111110101010", "1111111110101011", "1111111110101100", "1111111110101101", "11111010", "111111110111", "1111111110101110", "1111111110101111", "1111111110110000", "1111111110110001", "1111111110110010", "1111111110110011", "1111111110110100", "1111111110110101", "111111000", "111111111000000", "1111111110110110", "1111111110110111", "1111111110111000", "1111111110111001", "1111111110111010", "1111111110111011", "1111111110111100", "1111111110111101", "111111001", "1111111110111110", "1111111110111111", "1111111111000000", "1111111111000001", "1111111111000010", "1111111111000011", "1111111111000100", "1111111111000101", "1111111111000110", "111111010", "1111111111000111", "1111111111001000", "1111111111001001", "1111111111001010", "1111111111001011", "1111111111001100", "1111111111001101", "1111111111001110", "1111111111001111", "1111111001", "1111111111010000", "1111111111010001", "1111111111010010", "1111111111010011", "1111111111010100", "1111111111010101", "1111111111010110", "1111111111010111", "1111111111011000", "1111111010", "1111111111011001", "1111111111011010", "1111111111011011", "1111111111011100", "1111111111011101", "1111111111011110", "1111111111011111", "1111111111100000", "1111111111100001", "11111111000", "1111111111100010", "1111111111100011", "1111111111100100", "1111111111100101", "1111111111100110", "1111111111100111", "1111111111101000", "1111111111101001", "1111111111101010", "1111111111101011", "1111111111101100", "1111111111101101", "1111111111101110", "1111111111101111", "1111111111110000", "1111111111110001", "1111111111110010", "1111111111110011", "1111111111110100", "1111111111110101", "1111111111110110", "1111111111110111", "1111111111111000", "1111111111111001", "1111111111111010", "1111111111111011", "1111111111111100", "1111111111111101", "1111111111111110", "11111111001"]]
ac_chroma_table = [bitarray.bitarray(x) for x in ["00", "01", "100", "1010", "11000", "11001", "111000", "1111000", "111110100", "1111110110", "111111110100", "1011", "111001", "11110110", "111110101", "11111110110", "111111110101", "1111111110001000", "1111111110001001", "1111111110001010", "1111111110001011", "11010", "11110111", "1111110111", "111111110110", "111111111000010", "1111111110001100", "1111111110001101", "1111111110001110", "1111111110001111", "1111111110010000", "11011", "11111000", "1111111000", "111111110111", "1111111110010001", "1111111110010010", "1111111110010011", "1111111110010100", "1111111110010101", "1111111110010110", "111010", "111110110", "1111111110010111", "1111111110011000", "1111111110011001", "1111111110011010", "1111111110011011", "1111111110011100", "1111111110011101", "1111111110011110", "111011", "1111111001", "1111111110011111", "1111111110100000", "1111111110100001", "1111111110100010", "1111111110100011", "1111111110100100", "1111111110100101", "1111111110100110", "1111001", "11111110111", "1111111110100111", "1111111110101000", "1111111110101001", "1111111110101010", "1111111110101011", "1111111110101100", "1111111110101101", "1111111110101110", "1111010", "11111111000", "1111111110101111", "1111111110110000", "1111111110110001", "1111111110110010", "1111111110110011", "1111111110110100", "1111111110110101", "1111111110110110", "11111001", "1111111110110111", "1111111110111000", "1111111110111001", "1111111110111010", "1111111110111011", "1111111110111100", "1111111110111101", "1111111110111110", "1111111110111111", "111110111", "1111111111000000", "1111111111000001", "1111111111000010", "1111111111000011", "1111111111000100", "1111111111000101", "1111111111000110", "1111111111000111", "1111111111001000", "111111000", "1111111111001001", "1111111111001010", "1111111111001011", "1111111111001100", "1111111111001101", "1111111111001110", "1111111111001111", "1111111111010000", "1111111111010001", "111111001", "1111111111010010", "1111111111010011", "1111111111010100", "1111111111010101", "1111111111010110", "1111111111010111", "1111111111011000", "1111111111011001", "1111111111011010", "111111010", "1111111111011011", "1111111111011100", "1111111111011101", "1111111111011110", "1111111111011111", "1111111111100000", "1111111111100001", "1111111111100010", "1111111111100011", "11111111001", "1111111111100100", "1111111111100101", "1111111111100110", "1111111111100111", "1111111111101000", "1111111111101001", "1111111111101010", "1111111111101011", "1111111111101100", "11111111100000", "1111111111101101", "1111111111101110", "1111111111101111", "1111111111110000", "1111111111110001", "1111111111110010", "1111111111110011", "1111111111110100", "1111111111110101", "111111111000011", "1111111111110110", "1111111111110111", "1111111111111000", "1111111111111001", "1111111111111010", "1111111111111011", "1111111111111100", "1111111111111101", "1111111111111110", "1111111010"]]

# Decoder Class
class Decoder:
    def __init__(self, bit_array, image_size):
        # Setting up states
        self.channel = 'y'
        self.dc = True
        self.index = 0

        self.block_count = 0

        # Bit array containing Huffman-encoded JPEG-data
        self.bit_array = self.escape_ff(bit_array)
        # self.bit_array = bit_array

        # Image dimensions in a tuple (width, height)
        self.image_size = image_size

        self.__block_width = math.ceil(image_size[0] / 8)
        self.__block_height = math.ceil(image_size[1] / 8)

        print(self.__block_width)
        print(self.__block_height)

        # NP array containing unzigzagged image data (prior to inverting quantization, DCT, level shift and the color space transformation)
        self.blocks = np.array([])
        self.__row = np.array([])

    def escape_ff(self, bit_array):
        ba_bytes = bit_array.tobytes()

        out_bytes = bytearray()

        prev_byte_ff = False
        for i in range(len(ba_bytes)):
            current_byte = ba_bytes[i]

            if prev_byte_ff and current_byte == 0x00:
                prev_byte_ff = False
                continue

            out_bytes.append(current_byte)

            # If JPEG marker, check if next byte is 00
            if current_byte == 0xFF:
                prev_byte_ff = True
                if i == len(ba_bytes) - 1:
                    raise ValueError("0xFF was encountered in bitarray, without any byte stuffing.")
                
                next_byte = ba_bytes[i + 1]
                if next_byte != 0x00:
                    raise ValueError("0xFF was encountered in bitarray, without any byte stuffing.")
        result_ba = bitarray.bitarray()

        result_ba.frombytes(bytes(out_bytes))

        with open('REEE.bin', 'wb') as f:
            result_ba.tofile(f)

        return result_ba
    
    # Scans the bitarray for matching Huffman code in 'table'. 
    def scan_until_match(self, table, ac=False):
        # From current index to end of binary array
        for i in range(1, len(self.bit_array) - self.index + 1):
            # Slice binary array 
            sub_string = self.bit_array[self.index:self.index + i]
            
            # Try to find the sub_string in Huffman table specified by 'table'
            try:
                category_index = table.index(sub_string)

                self.index = self.index + i
                if ac:
                    if category_index == 0:
                        if DEBUG:
                            print("EOB")
                        return (0, 0)
                    elif category_index == len(table) - 1:
                        if DEBUG:
                            print("ZRL")
                        return (15, 0)
                    if DEBUG:
                        print(((category_index - 1) // 10, (category_index - 1) % 10 + 1))
                    return ((category_index - 1) // 10, (category_index - 1) % 10 + 1)
                else:

                    return category_index
            except ValueError:
                continue
        else:
            # If end of bit array is reached without finding a valid Huffman code, the string is malformed
            # print(self.index)
            # print(len(self.bit_array))
            if '0' in sub_string.to01():
                raise ValueError("NO VALID HUFFMAN CODE FOUND.")
            else:
                self.index = self.index + i
                return None

    def extract_amplitude(self, category):
        # If category is 0, there will be no amplitude to extract
        if not category:
            return None

        # Slice the bits containing the amplitude
        substring = self.bit_array[self.index:self.index + category]

        # If the first bit of the slice is 0, the number is a negative number represented in one's complement.
        if substring[0] == 0:
            return_value = -(ba2int(substring) ^ int('1' * category, 2))
        else:
            return_value = ba2int(substring)

        # Adjust index in bitarray to be after the amplitude-bits.
        self.index = self.index + category

        if self.block_count == Y_BLOCK_MIN * self.__block_width + X_BLOCK_MIN:
            print("AMPL.", return_value)

        return return_value

    def unzigzag(self, meta_block):
        a = meta_block[:, :, 0].flatten()
        b = meta_block[:, :, 1].flatten()
        c = meta_block[:, :, 2].flatten()

        ZIGZAG = np.array([[0,  1,  5,  6,  14, 15, 27, 28],
                   [2,  4,  7,  13, 16, 26, 29, 42],
                   [3,  8,  12, 17, 25, 30, 41, 43],
                   [9,  11, 18, 24, 31, 40, 44, 53],
                   [10, 19, 23, 32, 39, 45, 52, 54],
                   [20, 22, 33, 38, 46, 51, 55, 60],
                   [21, 34, 37, 47, 50, 56, 59, 61],
                   [35, 36, 48, 49, 57, 58, 62, 63]])

        # Turn ZIGZAG-matrix into 1D array that can be used to index into a 1D array.
        ZIGZAGFLAT = ZIGZAG.flatten()

        meta_block[:, :, 0] = a[ZIGZAGFLAT].reshape((8, 8))
        meta_block[:, :, 1] = b[ZIGZAGFLAT].reshape((8, 8))
        meta_block[:, :, 2] = c[ZIGZAGFLAT].reshape((8, 8))

        return meta_block

    def add_block(self, meta_block):
        # Convert Python list to NP array. Unzigzag the array. Reshape the 1D array into 2D matrix.
        new_block = np.copy(self.unzigzag(meta_block))

        self.block_count += 1

        if not self.__row.any():
            self.__row = new_block
            #print(self.__row.shape)
        else:
            # If row is not full yet, stack horizontally onto it
            self.__row = np.hstack((self.__row, new_block))
        if self.__row.shape[1] // 8 == self.__block_width:
            # If first row, initialize self.blocks, otherwise vertically stack onto the already-existing blocks.
            if not self.blocks.any():
                self.blocks = self.__row
            else:
                self.blocks = np.vstack((self.blocks, self.__row))
            
            self.__row = np.array([])

    def decode_dc(self, table):
        cat = self.scan_until_match(table)

        if cat == 0:
            amplitude = 0
        else:
            amplitude = self.extract_amplitude(cat)

        return amplitude

    def decode(self):
        global DEBUG
        meta_block = np.zeros((8, 8, 3), dtype=np.int16)
        block_values = []
        prev_cat = []

        


        while self.index < (len(self.bit_array) - 1):
            if self.index >= 0:
                DEBUG = True
            if self.index >= 100:
                DEBUG = False

            if self.dc:
                if self.channel == 'y':
                    # Find category and amplitude of DC-value
                    amp = self.decode_dc(dc_luma_table)

                    # Append DC-value to list of block values
                    block_values.append(amp)
                    
                    # Change state to AC
                    self.dc = False
                elif self.channel == 'cb':
                    amp = self.decode_dc(dc_chroma_table)
                    block_values.append(amp)
                    
                    self.dc = False
                elif self.channel == 'cr':
                    amp = self.decode_dc(dc_chroma_table)
                    block_values.append(amp)
                    
                    self.dc = False
                if DEBUG:
                    print(f"diff: {amp}, channel: {self.channel}, {self.index}")
            else:
                if self.channel == 'y':
                    cat = self.scan_until_match(ac_luma_table, ac=True)

                    # If EOB extend the length of list to 64, by appending 0
                    if cat:
                        if cat[0] == 0 and cat[1] == 0:
                            block_values.extend([0] * (64 - len(block_values)))
                        elif cat[0] == 15 and cat[1] == 0:
                            if DEBUG:
                                print("ZRRRRRRRL")
                            block_values.extend([0] * 16)
                            continue
                        else:
                            # If not EOB, extend by run-size amount of 0. Extract amplitude and append that value.
                            block_values.extend([0] * cat[0])
                            amplitude = self.extract_amplitude(cat[1])
                            #print("AMP", amplitude)
                            if amplitude:
                                block_values.append(amplitude)
                    else:
                        print("Not cat", self.block_count)
                        print(len(block_values))
                    #print(block_values)
                    if len(block_values) == 64:
                        # EOB reached, change to Cb DC. Create first layer of meta block.
                        self.dc = True
                        self.channel = 'cb'
                        meta_block[:, :, 0] = np.array(block_values).reshape((8, 8))
                        
                        block_values = []
                elif self.channel == 'cb':
                    cat = self.scan_until_match(ac_chroma_table, ac=True)
                    # If EOB extend the length of list to 64, by appending 0
                    if cat[0] == 0 and cat[1] == 0:
                        block_values.extend([0] * (64 - len(block_values)))
                    if cat[0] == 15 and cat[1] == 0:
                        block_values.extend([0] * 16)
                        continue
                    
                    # If not EOB, extend by run-size amount of 0. Extract amplitude and append that value.
                    block_values.extend([0] * cat[0])
                    amplitude = self.extract_amplitude(cat[1])

                    if amplitude:
                        block_values.append(amplitude)

                    if len(block_values) == 64:
                        # EOB reached, change to Cr DC. Create second layer of meta block.
                        self.dc = True
                        self.channel = 'cr'

                        try:
                            meta_block[:, :, 1] = np.array(block_values).reshape((8, 8))
                        except:
                            print(block_values)
                            exit()
                        
                        block_values = []
                elif self.channel == 'cr':
                    cat = self.scan_until_match(ac_chroma_table, ac=True)

                    # If EOB extend the length of list to 64, by appending 0
                    if cat:
                        if cat[0] == 0 and cat[1] == 0:
                            block_values.extend([0] * (64 - len(block_values)))
                            
                            self.dc = True
                            self.channel = 'y'

                            meta_block[:, :, 2] = np.array(block_values).reshape((8, 8))


                            self.add_block(meta_block)
                            block_values = []
                            prev_cat = []
                            continue
                    else:
                        print("Not cat", self.block_count)
                    if cat[0] == 15 and cat[1] == 0:
                        block_values.extend([0] * 16)
                        continue

                    # If not EOB, extend by run-size amount of 0. Extract amplitude and append that value.
                    block_values.extend([0] * cat[0])
                    amplitude = self.extract_amplitude(cat[1])

                    if amplitude:
                        block_values.append(amplitude)

                    if len(block_values) == 64:
                        # EOB reached, change to Y DC. Create third layer of meta block.
                        self.dc = True
                        self.channel = 'y'
                        meta_block[:, :, 2] = np.array(block_values).reshape((8, 8))

                        self.add_block(meta_block)
                        block_values = []
                        prev_cat = []

                    prev_cat.append(cat)
                if DEBUG:
                    print(f"Value: {amplitude}, cat: {cat[1]}, run: {cat[0]}, channel: {self.channel}, {self.index}")
        if int(self.blocks.shape[0] / 8) * int(self.blocks.shape[1] / 8) != self.__block_width * self.__block_height:
            print(f"Showing block x = ({X_BLOCK_MIN * 8}, {X_BLOCK_MAX * 8}) and y = ({Y_BLOCK_MIN * 8}, {Y_BLOCK_MAX * 8})")
            print(self.blocks[Y_BLOCK_MIN*8:Y_BLOCK_MAX*8, X_BLOCK_MIN*8:X_BLOCK_MAX*8, 0])
            print(self.blocks[Y_BLOCK_MIN*8:Y_BLOCK_MAX*8, X_BLOCK_MIN*8:X_BLOCK_MAX*8, 1])
            print(self.blocks[Y_BLOCK_MIN*8:Y_BLOCK_MAX*8, X_BLOCK_MIN*8:X_BLOCK_MAX*8, 2])

            print(self.blocks.shape[0], self.blocks.shape[1], self.__block_width, self.__block_height)
            # print(np.hsplit(self.blocks, 4))
            print(self.index, len(self.bit_array))
            print(self.block_count)


            raise ValueError("There aren't as many blocks as expected. The given dimensions are probably incorrect.")

if __name__ == '__main__':
    ba = bitarray.bitarray('')
    ba_full = bitarray.bitarray('110111101101001011100011000001110100111010001111111111100000010111111100110011010100110100101111100001111110011011111111001101100110000101001011111000011111100110111111110011011001110101000111100110100011111111001000101010100010000101110011101000101101001001111111111001000001110100010111111111000000101111111001100110101000101001011111000011111100110111111110011011001110011110100101101100111111010101111111100111110101100110001010101000100000')
    ba_correct = bitarray.bitarray('1010110010000101000101100001011010001100110001100100110010111011101110000110111101000001010')
    ba_small = bitarray.bitarray('1101111011010010111000110000011101001110100011111111111000000101111111001100110101001101001011111000011111100110111111110011011001100001010010111110000111111001101111111100110110011111')
    
    with open('source_of_truth_broken.bin', 'rb') as f:
        ba.fromfile(f)

    decoder = Decoder(ba, (1920, 1080))
    print(decoder.bit_array[:100])
    decoder.decode()
    print(decoder.blocks.dtype)
    print(decoder.blocks.shape)
    print(decoder._Decoder__row.shape)
    print(decoder.block_count)
    print("h", decoder._Decoder__block_height)
    print("w", decoder._Decoder__block_width)
    

    print(f"Showing block x = ({X_BLOCK_MIN * 8}, {X_BLOCK_MAX * 8}) and y = ({Y_BLOCK_MIN * 8}, {Y_BLOCK_MAX * 8})")
    print(decoder.blocks[Y_BLOCK_MIN*8:Y_BLOCK_MAX*8, X_BLOCK_MIN*8:X_BLOCK_MAX*8, 0].flatten())
    print(decoder.blocks[Y_BLOCK_MIN*8:Y_BLOCK_MAX*8, X_BLOCK_MIN*8:X_BLOCK_MAX*8, 1].flatten())
    print(decoder.blocks[Y_BLOCK_MIN*8:Y_BLOCK_MAX*8, X_BLOCK_MIN*8:X_BLOCK_MAX*8, 2].flatten())

    print(decoder.blocks.shape, decoder.blocks.size)
    print(decoder.blocks.dtype)
    decoder.blocks.flatten().tofile("nature_matrix.bin")

    # with open('plain_decoded_nature_clean.txt', 'w') as f:
    #     for y in range(decoder._Decoder__block_height):
    #         for x in range(decoder._Decoder__block_width):
    #             # f.write(f"Block: {x}, {y}\n")
    #             for i in range(3):
    #                 block = decoder.blocks[y*8:(y + 1)*8, x*8:(x + 1)*8, i].flatten()
    #                 line = ", ".join([str(x) for x in block])
    #                 f.write(f'{line}\n')

    #             # f.write('\n')

    # print(decoder.blocks[8:16, 0:8, 0])
    # print(decoder.blocks[8:16, 0:8, 1])
    # print(decoder.blocks[8:16, 0:8, 2])

    # print(decoder.blocks[8:16, 8:16, 0])
    # print(decoder.blocks[8:16, 8:16, 1])
    # print(decoder.blocks[8:16, 8:16, 2])
    
    

    # print(stacked.reshape((16, 16, 3)))
    # stacked_reshape = stacked.reshape((16, 16, 3), order='F')
    # print(stacked_reshape[0:8, 0:8, 0])

