import cv2 as cv
import numpy as np
import pyopencl as cl


def Setup():
    try:
        plaforms = cl.get_platforms()  # configure platform
        global plaform
        plaform = plaforms[0]

        devices = plaform.get_devices()  # configure device
        global device
        device = devices[0]

        global ctx  # set context
        ctx = cl.Context(devices)  # or dev_type=cl.device_type.ALL)
        global commQ
        commQ = cl.CommandQueue(ctx, device)  # create command queue

        file = open("Hough.cl", "r")  # load file with program/kernel

        global prog  # get the kernel
        prog = cl.Program(ctx, file.read())
        prog.build()  # build the program/kernel

    except Exception as e:
        print(e)
    return plaform, device, ctx, commQ, prog