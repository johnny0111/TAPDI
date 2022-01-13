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

def Circle_Ht(img, r_min, r_max,plaform, device, ctx, commQ, prog):

    # Define Image object: create image and buffer objects
    imgFormat = cl.ImageFormat(
        cl.channel_order.BGRA,
        cl.channel_type.UNSIGNED_INT8)


    accomulator = np.zeros(round(img.shape[0]*img.shape[1]*1000), dtype=int)
    maxval = np.int32(0)
    circle = np.zeros(3, dtype=int)
    width = img.shape[1]
    height = img.shape[0]

    # Buffer In
    bufferIn = cl.Image(
      ctx,
        flags=cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_ONLY,
        format=imgFormat,
        shape=(img.shape[1], img.shape[0]),  # image width, height
        pitches=(img.strides[0], img.strides[1]),
        hostbuf=img.data)

    # Buffer Out
    BufferAcc = cl.Buffer(
        ctx,
        flags = cl.mem_flags.COPY_HOST_PTR |
                cl.mem_flags.READ_WRITE,
        size = accomulator.nbytes,
        hostbuf = accomulator
    )

    BufferCircle= cl.Buffer(
        ctx,
        flags=cl.mem_flags.COPY_HOST_PTR |
              cl.mem_flags.READ_WRITE,
        size=circle.nbytes,
        hostbuf=circle
    )
    BufferMaxval= cl.Buffer(
        ctx,
        flags=cl.mem_flags.COPY_HOST_PTR |
              cl.mem_flags.READ_WRITE,
        hostbuf=maxval
    )

    # Setup Work items and groups
    dimension = 1  # R,G,B,A
    xBlockSize = 16
    yBlockSize = 16

    xBlocksNumber = round(img.shape[1] / xBlockSize)
    yBlocksNumber = round(img.shape[0] / yBlockSize)

    workItemSize = (xBlockSize, yBlockSize)
    workGroupSize = (xBlocksNumber * xBlockSize, yBlocksNumber * yBlockSize)

    kernelName = prog.hough_circle

    kernelName.set_arg(0, bufferIn)
    kernelName.set_arg(1, BufferAcc)
    kernelName.set_arg(2, np.int32(width))
    kernelName.set_arg(3, BufferCircle)
    kernelName.set_arg(4, np.int32(r_min))
    kernelName.set_arg(5, np.int32(r_max))
    kernelName.set_arg(6, BufferMaxval)
    kernelName.set_arg(7, np.int32(height))

    # Start program
    kernelEvent = cl.enqueue_nd_range_kernel(commQ, kernelName, global_work_size=workGroupSize,
                                         local_work_size=workItemSize)  # execute kernel
    kernelEvent.wait()  # wait for kernel to finish

    # Get results from device
    # cl.enqueue_copy(
    #     commQ,
    #     dest=accomulator.data,
    #     src=BufferAcc
    # )
    cl.enqueue_copy(
        commQ,
        dest=circle.data,
        src=BufferCircle
    )
    #
    # cl.enqueue_copy(
    #     commQ,
    #     dest=maxval.data,
    #     src=BufferMaxval
    # )


    bufferIn.release()
    BufferAcc.release()
    BufferCircle.release()
    BufferMaxval.release()
    return circle