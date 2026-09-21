import Foundation

/// Uses the production projection without inventing a real application identity.
@main
enum ProjectSyntheticContext {
    struct Input: Decodable {
        var applicationCategory: ApplicationCategory
        var fieldRole: String
        var fieldLabel: String
        var selectedText: String
        var surroundingText: String
        var hasAccessibility: Bool
        var isSecure: Bool
        var capture: Capture?
    }

    /// Authoring evidence only. This object is never a student input field.
    struct Capture: Decodable {
        var textWindow: String
        var selectionLocation: Int?
        var selectionLength: Int?
        var nearbyText: [String]

        func projected(_ input: Input) throws -> String {
            guard input.surroundingText.isEmpty,
                  textWindow.count <= 1_700, nearbyText.count <= 4,
                  nearbyText.allSatisfy({ $0.count <= 240 }),
                  nearbyText.reduce(0, { $0 + $1.count }) <= 600,
                  (selectionLocation == nil) == (selectionLength == nil) else {
                throw CocoaError(.coderInvalidValue)
            }
            let hasContent = !textWindow.isEmpty || !nearbyText.isEmpty || !input.selectedText.isEmpty
            guard (!hasContent || (input.hasAccessibility && !input.isSecure)),
                  input.hasAccessibility || selectionLocation == nil else {
                throw CocoaError(.coderInvalidValue)
            }
            var selection: NSRange?
            if let location = selectionLocation, let length = selectionLength {
                guard location >= 0, length >= 0,
                      location <= textWindow.utf16.count,
                      length <= textWindow.utf16.count - location,
                      length <= 1_200 else { throw CocoaError(.coderInvalidValue) }
                let candidate = NSRange(location: location, length: length)
                guard let range = FocusText.validRange(candidate, in: textWindow),
                      String(textWindow[range]) == input.selectedText else {
                    throw CocoaError(.coderInvalidValue)
                }
                selection = candidate
            } else if !input.selectedText.isEmpty {
                throw CocoaError(.coderInvalidValue)
            }
            return FocusText.render(textWindow: textWindow, selection: selection,
                                    selectedText: input.selectedText, nearbyText: nearbyText)
        }
    }

    static func main() throws {
        let decoder = JSONDecoder()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        while let line = readLine(strippingNewline: true) {
            guard line.utf8.count <= 1_048_576 else { throw CocoaError(.coderInvalidValue) }
            let input = try decoder.decode(Input.self, from: Data(line.utf8))
            // These cannot be obtained by PasteWhat's AX reader without trust.
            guard input.hasAccessibility || [input.fieldRole, input.fieldLabel, input.selectedText, input.surroundingText].allSatisfy(\.isEmpty) else {
                throw CocoaError(.coderInvalidValue)
            }
            let surrounding = try input.capture?.projected(input) ?? input.surroundingText
            let native = AppContext(appName: "", bundleID: "", fieldRole: input.fieldRole,
                                    fieldLabel: input.fieldLabel, selectedText: input.selectedText,
                                    surroundingText: surrounding,
                                    hasAccessibility: input.hasAccessibility, isSecure: input.isSecure)
            var projected = native.modelContext
            // Classification is supplied as controlled synthetic metadata. It
            // does not influence field/surface inference, and no fake bundle ID
            // or real host branding is introduced to make it fit.
            projected.applicationCategory = input.applicationCategory
            var data = try encoder.encode(projected)
            data.append(0x0A)
            try FileHandle.standardOutput.write(contentsOf: data)
        }
    }
}
