import pyopencl as cl
import imageForms as iF
import numpy as np
"""
platforms = cl.get_platforms()
for platform in platforms:
    name = platform.get_info(cl.platform_info.NAME)
    vendor = platform.get_info(cl.platform_info.VENDOR)
    version = platform.get_info(cl.platform_info.VERSION)
    displayStr = "Name: " + name + "\nVendor: " + vendor + "\nVersion: " + version + "\n"
    iF.showMessageBox(title="Platform Info", message=displayStr)
    devices = platform.get_devices()
    for device in devices:
        displayStr = "VENDOR: " + device.get_info(cl.device_info.VENDOR)
        displayStr = displayStr + "\nNAME: " + device.get_info(cl.device_info.NAME)
        displayStr = displayStr + "\nMAX_COMPUTE_UNITS: " + str(device.get_info(cl.device_info.MAX_COMPUTE_UNITS))
        displayStr = displayStr + "\nMAX_WORK_ITEM_DIMENSIONS: " + str(device.get_info(cl.device_info.MAX_WORK_ITEM_DIMENSIONS))
        displayStr = displayStr + "\nMAX_WORK_ITEM_SIZES: " + str(device.get_info(cl.device_info.MAX_WORK_ITEM_SIZES))
        displayStr = displayStr + "\nMAX_WORK_GROUP_SIZE: " + str(device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE))
        displayStr = displayStr + "\nMAX_CONSTANT_ARGS: " + str(device.get_info(cl.device_info.MAX_CONSTANT_ARGS))
        displayStr = displayStr + "\nIMAGE_SUPPORT: " + str(device.get_info(cl.device_info.IMAGE_SUPPORT))
        displayStr = displayStr + "\nIMAGE2D_MAX_WIDTH: " + str(device.get_info(cl.device_info.IMAGE2D_MAX_WIDTH))
        displayStr = displayStr + "\nIMAGE2D_MAX_HEIGHT: " + str(device.get_info(cl.device_info.IMAGE2D_MAX_HEIGHT))
        displayStr = displayStr + "\nLOCAL_MEM_SIZE: " + str(device.get_info(cl.device_info.LOCAL_MEM_SIZE))
        iF.showMessageBox(title="Device Info", message=displayStr)
"""

if __name__ == "__main__":


        platforms = cl.get_platforms()
        global platform
        platform = platforms[0]
        devices = platform.get_devices()
        global device
        device = devices[0]
        global ctx
        ctx = cl.Context(devices)  # or dev_type=cl.device_type.ALL)
        global commQ
        commQ = cl.CommandQueue(ctx, device)

        file = open("prog.cl","r")
        global prog
        prog = cl.Program(ctx,file.read())
        prog.build()

        arrayIn = np.array([1,2,3,4,5,6,7,8,9,10], dtype=int)
        const = 2
        kernelName = prog.multiply
        memBuffer = cl.Buffer(ctx,
            flags= cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_WRITE, hostbuf=arrayIn)

        kernelName.set_arg(0,memBuffer)
        kernelName.set_arg(1, np.int32(const))
        workGroupSize = (20, 1)
        workItemSize = (10, 1)
        kernelEvent = cl.enqueue_nd_range_kernel(commQ, kernelName,
                                                 global_work_size=workGroupSize, local_work_size=workItemSize)
        kernelEvent.wait()
        cl.enqueue_copy(commQ, arrayIn, memBuffer)
        print(arrayIn)
        memBuffer.release()