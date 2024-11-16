import MetalKit

func dupBuffer(_ buffer : MTLBuffer) -> MTLBuffer {
    let newBuffer = device.makeBuffer(length: buffer.length, options: [])!
    let commandBuffer = commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeBlitCommandEncoder()!
    encoder.copy(from: buffer, sourceOffset:0, to: newBuffer, destinationOffset: 0, size: buffer.length * MemoryLayout<UInt32>.stride)
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return newBuffer
}

// XXX Write buffer
// https://stackoverflow.com/questions/36120854/swift-writing-a-byte-stream-to-file
// and
// https://developer.apple.com/documentation/foundation/nsdata/1547231-datawithbytes?language=objc

func main() {
    let a = makeWidthHeightBuffer(width: 10, height: 20)
    let b = dupBuffer(a)
    let contents : UnsafeMutableRawPointer = b.contents()
    let p : UnsafeMutablePointer<UInt32> = contents.bindMemory(to: UInt32.self, capacity: 2 * MemoryLayout<UInt32>.stride)
    print(p[0], p[1])
    // Also note
    // https://stackoverflow.com/questions/42561558/update-contents-of-mtlbuffer-in-metal

    print(device.name)
    let height : Int = 6144
    let width : Int = 6144

    let z = Int(CommandLine.arguments[1])!
    let doubles = z

//    let seeds : [(Int, UInt8)] = [
//        (width / 2 + (height / 2) * width - z, 1),
//        (width / 2 + (height / 2) * width + z, 1)
//    ]

    let seeds : [(Int, UInt8)] = [
        (width / 2 + (height / 2) * width, 1)
    ]

//    let seeds : [(Int, UInt8)] = [
//        (0, 1)
//    ]

    let noseeds : [(Int, UInt8)] = []

    let rgbBuffer = makeRgbBuffer(width: width, height: height)
    let inputBuffer1 = makeBuffer(width: width, height: height, seeds: seeds)
    let outputBuffer = makeBuffer(width: width, height: height, seeds: noseeds)
    let widthHeightBuffer = makeWidthHeightBuffer(width: width, height: height)
    let countBuffer = makeCountBuffer(width: width, height: height)

    print("Start")
    for i in 0...doubles {
        print("Doubles to go", doubles - i)
//        symmetricStable(width: width, height: height,
//                        inputBuffer1: inputBuffer1, outputBuffer: outputBuffer,
//                        widthHeightBuffer: widthHeightBuffer)
        stableAndCount(width: width, height: height,
                       inputBuffer1: inputBuffer1, outputBuffer: outputBuffer,
                       widthHeightBuffer: widthHeightBuffer,
                       countBuffer: countBuffer)
        if i < doubles
        {
          twoTimes(width: width, height: height,
                    inputBuffer1: inputBuffer1,
                    widthHeightBuffer: widthHeightBuffer)
          twoTimesCount(width: width, height: height,
                        countBuffer: countBuffer,
                        widthHeightBuffer: widthHeightBuffer)
        }
    }

    saveImage(filename: String(format: "xxx.%d.png", z),
              width: width, height: height,
              inputBuffer1: inputBuffer1, rgbBuffer: rgbBuffer,
              widthHeightBuffer: widthHeightBuffer)

    saveCountImage(filename: String(format: "yyy.%d.png", z),
                   width: width, height: height,
                   countBuffer: countBuffer, rgbBuffer: rgbBuffer,
                   widthHeightBuffer: widthHeightBuffer)

    saveCountData(width: width, height: height, countBuffer: countBuffer)
}

main()
