import PIL
from PIL import Image
import numpy
import math
import tkinter as tk
from tkinter import filedialog
LSB_mask = 0xff


def WriteBytes(RLECodes):
    byte = 0
    checker = 0
    move = 0

    binWriter = bytearray()

    for code in RLECodes:
        RLECodeValue = code[0]
        RLECodeLen = code[1]

        if checker != 0:
            #shift = 12 - checker -
        žt: ", checker)

            binWriter.append((RLECodeValue  >> (move + checker)) & LSB_mask) #brez maske vrednost presega 256
            move = move + 8
        checker = move - RLECodeLen
        move = 0


    return binWriter


def CountBitEncodeLen(integer):
    if integer == 0:
        #print("Woopie")
        return 1
    elif integer < 0:
        integer = integer *(-1)
        return int((math.log(integer) /
                    math.log(2)) + 1);
    else:
        return int((math.log(integer) /
                    math.log(2)) + 1);


def RLEencode(block, encoded_list):
    pixel = 0;
    integer = 0
    block_element_count = 8 * 8

    while pixel < block_element_count - 1:
        if block[pixel] != 0:
            pixel = pixel + 1
            integer = block[pixel]
            if  pixel < block_element_count - 1 and block[pixel] == 0 :
                pixel = pixel + 1
                runSize = 1
                while block[pixel] == 0 and block_element_count - 1 > pixel:
                    pixel = pixel + 1
                    runSize = runSize + 1

                encoded, encoded_run_size = RLEOptions_SignedtoUnsigned('A', integer, runSize)
            else:
                encoded, encoded_run_size = RLEOptions_SignedtoUnsigned('C', integer, 0)
        elif  block[pixel] == 0:
            runSize = 1
            pixel = pixel + 1
            while block[pixel] == 0 and pixel < block_element_count - 1:
                runSize = runSize + 1
                if runSize > block_element_count:
                    runSize = block_element_count - 1

                pixel = pixel + 1


            encoded, encoded_run_size= RLEOptions_SignedtoUnsigned('B', 0, runSize)

        encoded_list.append((encoded, encoded_run_size))

    return encoded_list

def signedToUnsigned(signed_int):
    encode_len = CountBitEncodeLen(signed_int)
    unsigned_int = 0
    unsigned_int |= 1 << encode_len
    unsigned_int |= (1 << encode_len) - (signed_int*(-1))

    return unsigned_int

def RLEOptionA(AC,encode_len,runSize):
    optionA = 0
    optionA |= runSize << (1)
    optionA |= encode_len << (7)
    optionA |= AC << (11)

    encode_len = encode_len + 12

    return optionA, encode_len

def RLEOptionB(runSize):
    len = 7

    optionB = 0

    optionB |= runSize << 1

    return optionB, len

def RLEOptionC(AC,encode_len):
    optionC = 0

    optionC |= 1

    optionC |= encode_len << (1)
    optionC |= AC << (5)
    encode_len = encode_len + 6
    return optionC, encode_len

def RLEOptions_SignedtoUnsigned(RLEoption,number,runSize):
    AC = 0
    encode_len = 0
    if number < 0:
        number = signedToUnsigned(number)
        encode_len = CountBitEncodeLen(number)
        AC |= number
    else:
        encode_len = CountBitEncodeLen(number)
        AC |= number

    if RLEoption == 'A':
        return RLEOptionA(AC,encode_len,runSize)
    elif RLEoption == 'B':
        return RLEOptionB(runSize)
    elif RLEoption == 'C':
        return RLEOptionC(AC,encode_len)


def chunkify(img, block_width=8, block_height=8):
    shape = img.shape
    print("Image shape: ", shape)
    x_len = shape[0] // block_width
    y_len = shape[1] // block_height

    chunks = []
    x_indices = [i for i in range(0, shape[0] + 1, block_width)]
    y_indices = [i for i in range(0, shape[1] + 1, block_height)]

    shapes = list(zip(x_indices, y_indices))

    for i in range(len(shapes)):
        try:
            start_x = shapes[i][0]
            start_y = shapes[i][1]
            end_x = shapes[i + 1][0]
            end_y = shapes[i + 1][1]
            chunks.append(shapes[start_x:end_x][start_y:end_y])
        except IndexError:
            print('End of Array')

    return chunks


def checkIfDivisible(width, height):
    if (width % 8 == 0 and height % 8 == 0):
        return 1
    else:
        return 0


def padImage(width, height, filepath, img):
    new_width = 0
    new_height = 0
    while (height % 8 != 0):
        height = height + 1
    while (width % 8 != 0):
        width = width + 1

    print("Padding image")
    result = Image.new(img.mode, (width, height), (0, 0, 0))
    result.paste(img, (0, 0))
    result.save(filepath)

    img_new = Image.open(filepath)
    newWidth, newHeight = img_new.size

    return img_new

def CalculateFDCT(u, v, chunk):
    Cu = 0
    Cv = 0
    rgb_DCT = numpy.zeros((3), dtype=numpy.float32)

    if u == 0:
        Cu = 1 / math.sqrt(2)
    else:
        Cu = 1

    if v == 0:
        Cv = 1 / math.sqrt(2)
    else:
        Cv = 1

    for x in range(8):
        for y in range(8):
            rgb_DCT = rgb_DCT + chunk[x, y, :] * math.cos(((x * 2 + 1) * u * math.pi) / 16) * math.cos(
                ((y * 2 + 1) *  math.pi * v ) / 16)
    rgb_DCT = rgb_DCT * 0.25 * Cu * Cv
    return rgb_DCT

def arrayToZigZag(array, compression_rate):
    sortedIndicesArray = numpy.zeros((64, 3), dtype=numpy.int16)
    blockSize = 8 * 8
   # sortedIndices = [0, 1, 8, 16, 9, 2, 3, 11, 18, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33,
    #                 40, 48, 41, 36, 27, 20, 13, 6, 7, 14, 21, 28, 35, 41, 49, 56, 57, 50, 43,
     #                36, 29, 22, 17, 23, 30, 37, 44, 51, 58, 59, 52, 45,
      #               38, 31, 39, 43, 53, 60, 61, 54, 47, 55, 62, 63]

    sortedIndices = [0,1,8,16,9,2,3,10,17,24,32,25,18,11,4,5,12,19,26,33,
          40,48,41,34,27,20,13,6,7,14,21,28,35,42,49,56,57,50,43,
          36,29,22,15,23,30,37,44,51,58,59,52,45,
          38,31,39,46,53,60,61,54,47,55,62,63]

    for i in range(blockSize):
        if i < (blockSize - compression_rate):
            sortedIndicesArray[i, :] = array[sortedIndices[i], :]

    return sortedIndicesArray

filepath = filedialog.askopenfilename()

img = Image.open(filepath)
width, height = img.size
if checkIfDivisible(width, height) != 1:
    print("Gotta pad it, chief!")
    img = padImage(width, height, filepath, img)
width, height = img.size
img = numpy.asarray(img)
img_test = (img - 128).astype(numpy.int8)
compression_rate = input("Compression rate: ")

emptyChunk = numpy.zeros((height, width, 3), dtype=numpy.int16)
array_64_rgb_channels = numpy.zeros((64, 3), dtype=numpy.int16) # 8x8 = 64 -> 1 array = (r,g,b)
#print("img shape", img.shape[2])

writeToBin = []

for i in range(0,width,8):
    for j in range(0,height ,8):
        image_chunk = img_test[i:i+8,j:j+8,:]
        FDCTchunk = emptyChunk[i:i+8,j:j+8,:]
        for u in range(8):
            for v in range(8):
                FDCTchunk[u,v,:] = CalculateFDCT(u,v,image_chunk).astype(numpy.int16)
        for i in range(3):
            array_64_rgb_channels[:, i] = numpy.ravel(FDCTchunk[:, :, i])
        onedimensional_array = arrayToZigZag(array_64_rgb_channels, 6)
        for rgb in range(3):
            writeToBin = RLEencode(onedimensional_array[:, rgb], writeToBin)

encodedBytes = WriteBytes(writeToBin)
f = open("test1.bin",'bw')
f.write(encodedBytes)
f.close()

print("Compression Completed!")

