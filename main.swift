import MetalKit

func main() {
    print(device.name)
    let height : Int = 2048
    let width : Int = 2048

    let z = Int(CommandLine.arguments[1])!
    let doubles = z

//    let seeds : [(Int, UInt8)] = [
//        (width / 2 + (height / 2) * width - z, 1),
//        (width / 2 + (height / 2) * width + z, 1)
//    ]

//    let seeds : [(Int, UInt8)] = [
//        (width / 2 + (height / 2) * width, 1)
//    ]

    let seeds : [(Int, UInt8)] = [
        (0, 1)
    ]

    let noseeds : [(Int, UInt8)] = []

    let rgbBuffer = makeRgbBuffer(width:width, height:height)
    let inputBuffer1 = makeBuffer(width: width, height: height, seeds: seeds)
    let outputBuffer = makeBuffer(width: width, height: height, seeds: noseeds)
    let widthHeightBuffer = makeWidthHeightBuffer(width: width, height: height)

    print("Start")
    for i in 0...doubles-1 {
        print("Doubles to go", doubles - i)
        symmetricStable(width: width, height: height,
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
