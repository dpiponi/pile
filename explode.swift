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
let library = try! device.makeLibrary(filepath: "explode.metallib")

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
    init(kernelName: String) {
        pipelineState = makePipeline(kernelName: kernelName)
    }

    func exec(_ commandQueue : MTLCommandQueue,
               //pipelineState : MTLComputePipelineState,
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

//let pipelineState = makePipeline(kernelName: "explode")
//let rgbPipelineState = makePipeline(kernelName: "make_rgb")
//let twoPipelineState = makePipeline(kernelName: "two_times")
//let reducePipelineState = makePipeline(kernelName: "max_grains")

let explodeKernel = Kernel(kernelName: "explode")
let rgbKernel = Kernel(kernelName: "make_rgb")
let twoKernel = Kernel(kernelName: "two_times")
let reduceKernel = Kernel(kernelName: "max_grains")

// Threads
let w = explodeKernel.pipelineState.threadExecutionWidth
let h = explodeKernel.pipelineState.maxTotalThreadsPerThreadgroup / w

func pointerFromBuffer<T>(_ buffer : MTLBuffer, capacity: Int) -> UnsafeMutablePointer<T> {
    let contents : UnsafeMutableRawPointer = buffer.contents()
    return contents.bindMemory(to: T.self, capacity: capacity)
}

func maxGrains(inputBuffer: MTLBuffer, width: Int, height: Int) -> UInt32 {
    let w = explodeKernel.pipelineState.maxTotalThreadsPerThreadgroup
    let numThreadgroups = MTLSize(width: (width*height + w - 1) / w,
                                  height: 1,
                                  depth: 1)
    let threadsPerThreadgroup = MTLSize(width: w, height: 1, depth: 1)

    let result : [UInt32] = [0]
    let resultBuffer = makeBufferFrom(result)

    reduceKernel.exec(commandQueue,
                      buffers: [inputBuffer, resultBuffer],
                      numThreadGroups: numThreadgroups,
                      numThreadsPerThreadgroup: threadsPerThreadgroup,
                      sharedMemoryLen: w * MemoryLayout<Int32>.stride)

    return pointerFromBuffer(resultBuffer, capacity: 1)[0]
}

func twoTimes(width : Int, height : Int, inputBuffer1 : MTLBuffer,
              widthHeightBuffer : MTLBuffer) -> Void {
    let numThreadgroups = MTLSize(width: (width + w - 1) / w,
                                  height: (height + h - 1) / h,
                                  depth: 1)
    let threadsPerThreadgroup = MTLSize(width: w, height: h, depth: 1)

    twoKernel.exec(commandQueue,
                   buffers: [inputBuffer1, widthHeightBuffer],
                   numThreadGroups: numThreadgroups,
                   numThreadsPerThreadgroup: threadsPerThreadgroup)
}

func saveImage(filename : String, width : Int, height : Int,
               inputBuffer1: MTLBuffer,
               rgbBuffer : MTLBuffer,
               widthHeightBuffer: MTLBuffer) -> Void {
    let numThreadgroups = MTLSize(width: (width + w - 1) / w,
                                   height: (height + h - 1) / h,
                                   depth: 1)
    let threadsPerThreadgroup = MTLSize(width: w, height: h, depth: 1)

    rgbKernel.exec(commandQueue,
                   buffers: [inputBuffer1, widthHeightBuffer, rgbBuffer],
                   numThreadGroups: numThreadgroups,
                   numThreadsPerThreadgroup: threadsPerThreadgroup)

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
