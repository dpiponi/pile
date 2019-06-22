import MetalKit

func stable(width : Int, height : Int,
            inputBuffer1 : MTLBuffer, outputBuffer : MTLBuffer,
            widthHeightBuffer : MTLBuffer) -> Void {
    let numThreadgroups = MTLSize(width: (width + w - 1) / w,
                                   height: (height + h - 1) / h,
                                   depth: 1)
    let threadsPerThreadgroup = MTLSize(width: w, height: h, depth: 1)

    while true {
        // Needs to be an even length loop
        for t in 0...255 {
            let buffers = (t % 2 == 0
                ? [inputBuffer1, widthHeightBuffer, outputBuffer]
                : [outputBuffer, widthHeightBuffer, inputBuffer1])

            exec(commandQueue,
                pipelineState: pipelineState,
                buffers: buffers,
                numThreadGroups: numThreadgroups,
                numThreadsPerThreadgroup: threadsPerThreadgroup);
        }

        let tallest = maxGrains(inputBuffer: inputBuffer1, width: width, height: height)
        if (tallest < 4) {
          return
        }
    }
}

func main() {
    let height : Int = 768
    let width : Int = 768
    let doubles = 18;

    let z = Int(CommandLine.arguments[1])!

//    let seeds : [(Int, UInt8)] = [
//        (width / 2 + (height / 2) * width - z, 1),
//        (width / 2 + (height / 2) * width + z, 1)
//    ]

    let seeds : [(Int, UInt8)] = [
        (width / 2 + (height / 2) * width, 1)
    ]
    let noseeds : [(Int, UInt8)] = []

    let rgbBuffer = makeRgbBuffer(width:width, height:height)
    let inputBuffer1 = makeBuffer(width: width, height: height, seeds: seeds)
    let outputBuffer = makeBuffer(width: width, height: height, seeds: noseeds)
    let widthHeightBuffer = makeWidthHeightBuffer(width: width, height: height)

    print("Start")
    for i in 0...doubles-1 {
        print("Doubles to go", doubles - i)
        stable(width: width, height: height,
               inputBuffer1: inputBuffer1, outputBuffer: outputBuffer,
               widthHeightBuffer: widthHeightBuffer)
        twoTimes(width: width, height: height,
                  inputBuffer1: inputBuffer1,
                  widthHeightBuffer: widthHeightBuffer)
    }

    saveImage(filename: String(format: "xxx.%d.png", z),
              width: width, height: height,
              inputBuffer1: inputBuffer1, rgbBuffer: rgbBuffer,
              widthHeightBuffer: widthHeightBuffer)
}

main()