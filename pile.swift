// Useful links:
//
// https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
// https://developer.apple.com/documentation/metal/calculating_threadgroup_and_grid_sizes
// https://developer.apple.com/documentation/metal/setting_up_a_command_structure
//
// wakita's minimal example
// https://gist.github.com/wakita/f4915757c6c6c128c05c8680cd859e1a
// 
// mateuszbuda's reduction code
// https://github.com/mateuszbuda/GPUExample/blob/master/GPUExample/kernel.metal

import MetalKit

let device = MTLCreateSystemDefaultDevice()!
let commandQueue = device.makeCommandQueue()!
let library = try! device.makeLibrary(filepath: "pile.metallib")

func makePipeline(kernelName: String) -> MTLComputePipelineState {
    return try! device.makeComputePipelineState(
        function: library.makeFunction(name: kernelName)!)
}

func makeBufferFrom<T>(_ data : [T]) -> MTLBuffer {
    return device.makeBuffer(bytes: data,
                             length: MemoryLayout<T>.stride * data.count,
                             options: [])!
}

func makeBuffer<T : Numeric>(width : Int, height : Int, seeds: [(Int, T)] = []) -> MTLBuffer {
  let zero : T = 0
  var input: [T] = Array(repeating: zero, count: width * height)
  for (i, z) in seeds {
    input[i] = z
  }
  return makeBufferFrom(input)
}

func makeRgbBuffer(width : Int, height : Int) -> MTLBuffer {
    return device.makeBuffer(length: MemoryLayout<UInt8>.stride * 3 * width * height,
                             options: [])!
}

func makeWidthHeightBuffer(width : Int, height : Int) -> MTLBuffer {
    let input2 : [UInt32] = [UInt32(height), UInt32(width)]
    return makeBufferFrom(input2)
}

let result : [UInt32] = [0]
let resultBuffer = makeBufferFrom(result)

class Kernel {
    var pipelineState : MTLComputePipelineState
    var w : Int
    var h : Int

    init(kernelName: String) {
        pipelineState = makePipeline(kernelName: kernelName)

        w = pipelineState.threadExecutionWidth
        h = pipelineState.maxTotalThreadsPerThreadgroup / w
    }

    func exec(_ commandQueue : MTLCommandQueue,
               buffers : [MTLBuffer],
               numThreadGroups : MTLSize,
               numThreadsPerThreadgroup : MTLSize,
               sharedMemoryLen : Int = 0) -> Void {
        let commandBuffer = commandQueue.makeCommandBuffer()!
        let encoder = commandBuffer.makeComputeCommandEncoder()!

        encoder.setComputePipelineState(pipelineState)
        for i in 0...buffers.count-1 {
            encoder.setBuffer(buffers[i], offset: 0, index: i)
        }
        if (sharedMemoryLen > 0) {
            encoder.setThreadgroupMemoryLength(sharedMemoryLen, index: 0)
        }

        encoder.dispatchThreadgroups(numThreadGroups,
                                     threadsPerThreadgroup: numThreadsPerThreadgroup)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
}

class Kernel1d: Kernel {
    func exec1d(_ commandQueue : MTLCommandQueue,
               buffers : [MTLBuffer],
               width: Int,
               height: Int,
               sharedMemoryLen : Int = 0) -> Void {

        let numThreadgroups = MTLSize(width: (width*height + w - 1) / w,
                                      height: 1, depth: 1)
        let threadsPerThreadgroup = MTLSize(width: w, height: 1, depth: 1)

        exec(commandQueue,
             buffers: buffers,
             numThreadGroups: numThreadgroups,
             numThreadsPerThreadgroup: threadsPerThreadgroup,
             sharedMemoryLen: sharedMemoryLen);
    }
}

class Kernel2d: Kernel {
    func exec2d(_ commandQueue : MTLCommandQueue,
               buffers : [MTLBuffer],
               width: Int,
               height: Int,
               sharedMemoryLen : Int = 0) -> Void {

        let numThreadgroups = MTLSize(width: (width + w - 1) / w,
                                       height: (height + h - 1) / h,
                                       depth: 1)
        let threadsPerThreadgroup = MTLSize(width: w, height: h, depth: 1)

        exec(commandQueue,
             buffers: buffers,
             numThreadGroups: numThreadgroups,
             numThreadsPerThreadgroup: threadsPerThreadgroup,
             sharedMemoryLen: sharedMemoryLen);
    }
}

let explodeKernel = Kernel2d(kernelName: "explode")
let symmetricExplodeKernel = Kernel2d(kernelName: "symmetric_explode")
let twoKernel = Kernel2d(kernelName: "two_times")
let rgbKernel = Kernel2d(kernelName: "make_rgb")
let reduceKernel = Kernel1d(kernelName: "max_grains")

func stable(width : Int, height : Int,
            inputBuffer1 : MTLBuffer, outputBuffer : MTLBuffer,
            widthHeightBuffer : MTLBuffer) -> Void {

    while true {
        // Needs to be an even length loop
        for t in 0...255 {
            let buffers = (t % 2 == 0
                ? [inputBuffer1, widthHeightBuffer, outputBuffer]
                : [outputBuffer, widthHeightBuffer, inputBuffer1])

            explodeKernel.exec2d(commandQueue,
                                 buffers: buffers,
                                 width: width, height: height);
        }

        let tallest = maxGrains(inputBuffer: inputBuffer1, width: width, height: height)
        if (tallest < 4) {
          return
        }
    }
}

func symmetricStable(width : Int, height : Int,
                     inputBuffer1 : MTLBuffer, outputBuffer : MTLBuffer,
                     widthHeightBuffer : MTLBuffer) -> Void {

    while true {
        // Needs to be an even length loop
        for t in 0...255 {
            let buffers = (t % 2 == 0
                ? [inputBuffer1, widthHeightBuffer, outputBuffer]
                : [outputBuffer, widthHeightBuffer, inputBuffer1])

            symmetricExplodeKernel.exec2d(commandQueue,
                                          buffers: buffers,
                                          width: width, height: height);
        }

        let tallest = maxGrains(inputBuffer: inputBuffer1, width: width, height: height)
        if (tallest < 4) {
          return
        }
    }
}

func pointerFromBuffer<T>(_ buffer : MTLBuffer, capacity: Int) -> UnsafeMutablePointer<T> {
    let contents : UnsafeMutableRawPointer = buffer.contents()
    return contents.bindMemory(to: T.self, capacity: capacity)
}

func maxGrains(inputBuffer: MTLBuffer, width: Int, height: Int) -> UInt32 {
    let result : [UInt32] = [0]
    let resultBuffer = makeBufferFrom(result)

    reduceKernel.exec1d(commandQueue,
                        buffers: [inputBuffer, resultBuffer],
                        width: width, height: height,
                        sharedMemoryLen: reduceKernel.w * MemoryLayout<Int32>.stride)

    return pointerFromBuffer(resultBuffer, capacity: 1)[0]
}

func twoTimes(width : Int, height : Int, inputBuffer1 : MTLBuffer,
              widthHeightBuffer : MTLBuffer) -> Void {

    twoKernel.exec2d(commandQueue,
                     buffers: [inputBuffer1, widthHeightBuffer],
                     width: width, height: height)
}

func saveImage(filename : String, width : Int, height : Int,
               inputBuffer1: MTLBuffer,
               rgbBuffer : MTLBuffer,
               widthHeightBuffer: MTLBuffer) -> Void {

    rgbKernel.exec2d(commandQueue,
                     buffers: [inputBuffer1, widthHeightBuffer, rgbBuffer],
                     width: width, height: height)

    let pixelData : UnsafeMutablePointer<UInt8> = pointerFromBuffer(rgbBuffer, capacity: width * height * 3)

    let rgbData = CFDataCreate(nil, pixelData, 3*width*height)!
    let provider = CGDataProvider(data: rgbData)!
    let image = CGImage(width: width, height: height,
                        bitsPerComponent: 8, bitsPerPixel: 24, bytesPerRow: 3*width,
                        space: CGColorSpaceCreateDeviceRGB(),
                        bitmapInfo: CGBitmapInfo.byteOrderMask,
                        provider: provider,
                        decode: nil,
                        shouldInterpolate: true,
                        intent: CGColorRenderingIntent.defaultIntent)!

    let destination = CGImageDestinationCreateWithURL(
        CFURLCreateWithFileSystemPath(nil, filename as CFString,
                                      CFURLPathStyle.cfurlposixPathStyle, false), kUTTypePNG, 1, nil)!
    CGImageDestinationAddImage(destination, image, nil)
    CGImageDestinationFinalize(destination)
    print("Saved to", filename)
}