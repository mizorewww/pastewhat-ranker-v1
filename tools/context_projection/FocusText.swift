import Foundation

/// Encodes only text and selection boundaries actually obtained from AX.
/// Shared verbatim with synthetic-data preparation before teacher labeling.
enum FocusText {
    static let format = "pastewhat-focus-v1"

    static func render(textWindow: String, selection: NSRange?, selectedText: String,
                       nearbyText: [String], hostName: String = "") -> String {
        let window = String(textWindow.prefix(1_700))
        let nearby = normalizedNearby(nearbyText, hostName: hostName)
        var fields = ["\"format\":\"\(format)\""]
        if let selection, let range = validRange(selection, in: window),
           String(window[range]) == selectedText {
            fields.append("\"selectionKnown\":true")
            fields.append("\"beforeSelection\":\(json(String(window[..<range.lowerBound])))")
            fields.append("\"afterSelection\":\(json(String(window[range.upperBound...])))")
        } else {
            fields.append("\"selectionKnown\":false")
            fields.append("\"textWindow\":\(json(window))")
        }
        fields.append("\"nearbyText\":\(json(nearby))")
        // No synthetic scaffolding may turn absent context into semantic evidence.
        if window.isEmpty && nearby.isEmpty && selectedText.isEmpty { return "" }
        return "{" + fields.joined(separator: ",") + "}"
    }

    static func validRange(_ selection: NSRange, in text: String) -> Range<String.Index>? {
        let units = Array(text.utf16)
        guard selection.location >= 0, selection.length >= 0,
              selection.location <= units.count,
              selection.length <= units.count - selection.location else { return nil }
        for boundary in [selection.location, selection.location + selection.length] {
            // AX offsets are UTF-16. Splitting a surrogate pair is not a real
            // insertion point, even if Swift can represent the intermediate index.
            if boundary < units.count, (0xDC00...0xDFFF).contains(units[boundary]) { return nil }
        }
        return Range(selection, in: text)
    }

    static func normalizedNearby(_ values: [String], hostName: String) -> [String] {
        var result: [String] = []
        var remaining = 600
        let brand = hostName.isEmpty ? nil : "(?<![\\p{L}\\p{N}])"
            + NSRegularExpression.escapedPattern(for: hostName) + "(?![\\p{L}\\p{N}])"
        for value in values.prefix(4) {
            var label = value
            if let brand {
                label = label.replacingOccurrences(of: brand, with: "", options: [.caseInsensitive, .regularExpression])
            }
            label = String(label.trimmingCharacters(in: .whitespacesAndNewlines).prefix(min(240, remaining)))
            guard !label.isEmpty, !result.contains(label) else { continue }
            result.append(label)
            remaining -= label.count
            if remaining == 0 { break }
        }
        return result
    }

    private static func json<T: Encodable>(_ value: T) -> String {
        // String/array encoding cannot fail; the encoder escapes user-controlled
        // quotes and delimiters rather than treating them as boundary markers.
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes]
        guard let data = try? encoder.encode(value), let encoded = String(data: data, encoding: .utf8) else { return "null" }
        return encoded
    }
}
