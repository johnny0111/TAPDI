import cv2 as cv
import pyopencl as cl
import numpy as np
import imageForms as iF
import math

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

        file = open("brightness.cl", "r")  # load file with program/kernel

        global prog  # get the kernel
        prog = cl.Program(ctx, file.read())
        prog.build()  # build the program/kernel

    except Exception as e:
        print(e)
    return plaform, device, ctx, commQ, prog


def GPU(img, c, b, plaform, device, ctx, commQ, prog):
    # Convert to BGRA
    imageBGRA = cv.cvtColor(img, cv.COLOR_BGR2BGRA)
    imageBGRA_original = imageBGRA.copy()

    # Define Image object: create image and buffer objects
    imgFormat = cl.ImageFormat(
        cl.channel_order.BGRA,
        cl.channel_type.UNSIGNED_INT8)

    # Buffer In
    bufferIn = cl.Image(
      ctx,
        flags=cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_ONLY,
        format=imgFormat,
        shape=(imageBGRA.shape[1], imageBGRA.shape[0]),  # image width, height
        pitches=(imageBGRA.strides[0], imageBGRA.strides[1]),
        hostbuf=imageBGRA.data)

    # Buffer Out
    bufferOut = cl.Image(
        ctx,
        flags=cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.WRITE_ONLY,  # out is WRITE_ONLY
        format=imgFormat,
        shape=(imageBGRA.shape[1], imageBGRA.shape[0]),
        pitches=(imageBGRA.strides[0], imageBGRA.strides[1]),
        hostbuf=imageBGRA_original.data)

    # Other parameters
    w_in = imageBGRA.shape[1]
    h_in = imageBGRA.shape[0]

    # Setup Work items and groups
    dimension = 1  # R,G,B,A
    xBlockSize = 16
    yBlockSize = 16

    xBlocksNumber = round(imageBGRA.shape[1] / xBlockSize)
    yBlocksNumber = round(imageBGRA.shape[0] / yBlockSize)

    workItemSize = (xBlockSize, yBlockSize)
    workGroupSize = (xBlocksNumber * xBlockSize, yBlocksNumber * yBlockSize)

    # Send parameters to device
    kernelName = prog.sobel_BGRA

    kernelName.set_arg(0, bufferIn)  # set kernel/function arguments. 0=parameter number
    kernelName.set_arg(1, bufferOut)
    kernelName.set_arg(2, np.int32(w_in))
    kernelName.set_arg(3, np.int32(h_in))
    kernelName.set_arg(4, np.int32(b))
    kernelName.set_arg(5, np.float32(c))

    # Start program
    kernelEvent = cl.enqueue_nd_range_kernel(commQ, kernelName, global_work_size=workGroupSize,
                                         local_work_size=workItemSize)  # execute kernel
    kernelEvent.wait()  # wait for kernel to finish

    # Get results from device
    cl.enqueue_copy(
        commQ,
        dest=imageBGRA_original.data,
        src=bufferOut,
        origin=(0, 0),
        region=(imageBGRA.shape[1], imageBGRA.shape[0]),
        is_blocking=True
    )

    # Release device memory
    bufferOut.release()
    bufferIn.release()

    return imageBGRA_original

