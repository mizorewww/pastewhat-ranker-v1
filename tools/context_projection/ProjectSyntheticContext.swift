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
            let native = AppContext(appName: "", bundleID: "", fieldRole: input.fieldRole,
                                    fieldLabel: input.fieldLabel, selectedText: input.selectedText,
                                    surroundingText: input.surroundingText,
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
