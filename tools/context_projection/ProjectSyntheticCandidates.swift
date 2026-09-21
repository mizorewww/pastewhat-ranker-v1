import Foundation
import CoreGraphics
import ImageIO

/// Controlled authoring fixtures, projected through the exact production codec.
/// This command never touches the system pasteboard or reads a file URL.
@main
enum ProjectSyntheticCandidates {
    struct Input: Decodable {
        var id: String
        var sourceCategory: ApplicationCategory
        var payload: Payload
    }

    struct Payload: Decodable {
        var type: String
        var text: String?
        var names: [String]?
        var width: Int?
        var height: Int?

        func representations() throws -> [PasteboardPayload] {
            switch type {
            case "text":
                guard let text, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                      text.utf8.count <= 128_000 else { throw CocoaError(.coderInvalidValue) }
                return [PasteboardPayload(representations: ["public.utf8-plain-text": Data(text.utf8)])]
            case "file":
                guard let names, (1...20).contains(names.count), Set(names).count == names.count else {
                    throw CocoaError(.coderInvalidValue)
                }
                return try names.map { name in
                    guard !name.isEmpty, ![".", ".."].contains(name), name.utf8.count <= 255,
                          name.rangeOfCharacter(from: CharacterSet(charactersIn: "/\\\0\n\r")) == nil else {
                        throw CocoaError(.coderInvalidValue)
                    }
                    let url = URL(fileURLWithPath: "/PasteWhat-Synthetic/", isDirectory: true)
                        .appendingPathComponent(name, isDirectory: false)
                    return PasteboardPayload(representations: ["public.file-url": Data(url.absoluteString.utf8)])
                }
            case "image":
                guard let width, let height, (1...8192).contains(width), (1...8192).contains(height),
                      width * height <= 16_777_216 else { throw CocoaError(.coderInvalidValue) }
                // Blank grayscale pixels carry no invented image semantics. The
                // candidate's dimensions are recovered from the resulting PNG.
                let pixels = Data(repeating: 0, count: width * height)
                guard let provider = CGDataProvider(data: pixels as CFData),
                      let image = CGImage(width: width, height: height, bitsPerComponent: 8,
                                          bitsPerPixel: 8, bytesPerRow: width,
                                          space: CGColorSpaceCreateDeviceGray(),
                                          bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                                          provider: provider, decode: nil, shouldInterpolate: false,
                                          intent: .defaultIntent) else { throw CocoaError(.coderInvalidValue) }
                let bytes = NSMutableData()
                guard let destination = CGImageDestinationCreateWithData(bytes, "public.png" as CFString, 1, nil) else {
                    throw CocoaError(.coderInvalidValue)
                }
                CGImageDestinationAddImage(destination, image, nil)
                guard CGImageDestinationFinalize(destination) else { throw CocoaError(.coderInvalidValue) }
                return [PasteboardPayload(representations: ["public.png": bytes as Data])]
            default:
                throw CocoaError(.coderInvalidValue)
            }
        }
    }

    static func main() throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        while let line = readLine(strippingNewline: true) {
            guard line.utf8.count <= 8_388_608 else { throw CocoaError(.coderInvalidValue) }
            let inputs = try JSONDecoder().decode([Input].self, from: Data(line.utf8))
            guard (1...20).contains(inputs.count), Set(inputs.map(\.id)).count == inputs.count else {
                throw CocoaError(.coderInvalidValue)
            }
            let candidates = try inputs.map { input in
                let value = CandidateProjection.project(try input.payload.representations())
                return RecommendationCandidate(id: input.id, text: value.text, kind: value.kind.rawValue,
                                               capabilities: value.capabilities, sourceCategory: input.sourceCategory)
            }
            var data = try encoder.encode(candidates)
            data.append(0x0A)
            try FileHandle.standardOutput.write(contentsOf: data)
        }
    }
}
