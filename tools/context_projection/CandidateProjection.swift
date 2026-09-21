import Foundation
import ImageIO

/// The observable candidate view shared by clipboard capture and synthetic data.
/// Callers supply captured, supported payloads; no application state is consulted.
enum CandidateProjection {
    struct Summary: Sendable {
        var text: String
        var kind: ClipKind
        var capabilities: [String]
    }

    private static let utf8TextTypes = [
        "public.utf8-plain-text", "public.plain-text", "public.utf8-tab-separated-values-text", "public.url"
    ]
    private static let utf16TextTypes = ["public.utf16-external-plain-text", "public.utf16-plain-text"]
    private static let richTextTypes = ["public.rtf", "com.apple.flat-rtfd", "public.html"]
    private static let rasterImageTypes = ["public.png", "public.tiff"]
    private static let imageTypes = rasterImageTypes + ["com.adobe.pdf"]

    static let supportedTypes = Set(utf8TextTypes + utf16TextTypes + richTextTypes + imageTypes
                                   + ["public.file-url", "com.apple.cocoa.pasteboard.color"])

    static func project(_ payloads: [PasteboardPayload]) -> Summary {
        let types = Set(payloads.flatMap { $0.representations.keys }).intersection(supportedTypes)
        let description: (text: String, kind: ClipKind)
        if types.contains("public.file-url") {
            let names = payloads.compactMap { payload -> String? in
                guard let data = payload.representations["public.file-url"],
                      let string = String(data: data, encoding: .utf8), let url = URL(string: string) else { return nil }
                return url.lastPathComponent
            }
            description = (String(names.joined(separator: "\n").prefix(32_000)), .file)
        } else {
            let text = payloads.compactMap(Self.text(from:)).joined(separator: "\n")
            if !text.isEmpty {
                description = (String(text.prefix(32_000)), kind(for: text))
            } else if !types.isDisjoint(with: imageTypes) {
                description = (imageSummary(payloads), .image)
            } else if types.contains("com.apple.cocoa.pasteboard.color") {
                description = ("颜色", .color)
            } else {
                description = ("富文本", .text)
            }
        }

        var capabilities: [String] = []
        if !types.isDisjoint(with: utf8TextTypes + utf16TextTypes) { capabilities.append("text") }
        if !types.isDisjoint(with: imageTypes) { capabilities.append("image") }
        if types.contains("public.file-url") { capabilities.append("file") }
        if !types.isDisjoint(with: richTextTypes) {
            capabilities.append("richText")
            if !capabilities.contains("text") { capabilities.append("text") }
        }
        // The established protocol has no separate native-color capability.
        // Keep its generic fallback; the summary claims no unobserved color value.
        if capabilities.isEmpty { capabilities = ["text"] }
        return Summary(text: description.text, kind: description.kind, capabilities: capabilities)
    }

    static func text(from payload: PasteboardPayload) -> String? {
        for type in utf8TextTypes + ["public.file-url"] {
            if let data = payload.representations[type], let text = String(data: data, encoding: .utf8) {
                return text
            }
        }
        for type in utf16TextTypes {
            if let data = payload.representations[type], let text = String(data: data, encoding: .utf16) {
                return text
            }
        }
        return nil
    }

    // Preserve the deployed classifier exactly. A semantic command/code label
    // supplied by an author must not replace this observable classification.
    static func kind(for text: String) -> ClipKind {
        let value = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if value.range(of: "^#[0-9a-fA-F]{3,8}$", options: .regularExpression) != nil { return .color }
        if value.range(of: "^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$", options: .regularExpression) != nil { return .email }
        if let url = URL(string: value), ["https", "http", "ftp", "mailto"].contains(url.scheme?.lowercased() ?? ""),
           !value.contains(where: \.isWhitespace) { return .url }
        if value.range(of: "^[+()0-9 .-]{7,25}$", options: .regularExpression) != nil,
           value.filter(\.isNumber).count >= 7 { return .phone }
        if ["$ ", "git ", "sudo ", "npm ", "npx ", "brew ", "python ", "python3 ", "curl ",
            "swift ", "cd ", "ls ", "ssh ", "docker ", "make ", "xcodebuild "].contains(where: value.hasPrefix) { return .command }
        if ["import ", "func ", "let ", "const ", "def ", "class ", "struct ", "SELECT ", "#!/"].contains(where: value.hasPrefix)
            || (value.contains("\n") && (value.contains("{") || value.contains("=>"))) { return .code }
        return .text
    }

    private struct PixelSize: Equatable {
        var width: Int
        var height: Int

        var description: String { "\(width) × \(height) 像素" }
    }

    private static func imageSummary(_ payloads: [PasteboardPayload]) -> String {
        let images = payloads.filter { !Set($0.representations.keys).isDisjoint(with: imageTypes) }
        if images.count == 1 {
            return imageSize(images[0]).map { "图片 · \($0.description)" } ?? "图片"
        }
        let knownSizes = images.enumerated().compactMap { index, payload -> String? in
            imageSize(payload).map { "图片 \(index + 1) · \($0.description)" }
        }
        return (["\(images.count) 张图片"] + knownSizes).joined(separator: "\n")
    }

    private static func imageSize(_ payload: PasteboardPayload) -> PixelSize? {
        // A PDF has no intrinsic raster dimensions. If it is an alternative
        // representation, do not imply that every paste target gets these pixels.
        guard payload.representations["com.adobe.pdf"] == nil else { return nil }
        var result: PixelSize?
        for type in rasterImageTypes {
            guard let data = payload.representations[type] else { continue }
            guard let size = imageSize(data, declaredType: type) else { return nil }
            if let result, result != size { return nil }
            result = size
        }
        return result
    }

    private static func imageSize(_ data: Data, declaredType: String) -> PixelSize? {
        let options = [kCGImageSourceShouldCache: false] as CFDictionary
        guard let source = CGImageSourceCreateWithData(data as CFData, options),
              CGImageSourceGetType(source) as String? == declaredType,
              CGImageSourceGetCount(source) == 1,
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, options) as? [CFString: Any],
              CGImageSourceGetStatus(source) == .statusComplete,
              CGImageSourceGetStatusAtIndex(source, 0) == .statusComplete,
              let width = positiveInteger(properties[kCGImagePropertyPixelWidth]),
              let height = positiveInteger(properties[kCGImagePropertyPixelHeight]) else { return nil }
        let orientation: Int
        if let rawOrientation = properties[kCGImagePropertyOrientation] {
            guard let value = positiveInteger(rawOrientation), (1...8).contains(value) else { return nil }
            orientation = value
        } else {
            orientation = 1
        }
        // EXIF/TIFF orientations 5–8 rotate the displayed axes by 90 degrees.
        return orientation >= 5 ? PixelSize(width: height, height: width) : PixelSize(width: width, height: height)
    }

    private static func positiveInteger(_ value: Any?) -> Int? {
        guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
              let integer = Int(exactly: number.doubleValue), integer > 0 else { return nil }
        return integer
    }
}
