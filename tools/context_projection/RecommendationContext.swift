import Foundation

enum ApplicationCategory: String, Codable, Sendable {
    case browser, development, terminal, mail, messaging, writing, spreadsheet, creative
    case fileManagement = "file_management"
    case unknown

    static func classify(bundleID: String) -> Self {
        // Identity is resolved locally. Unknown applications never inherit a guessed content type.
        let id = bundleID.lowercased()
        let groups: [(Self, [String])] = [
            (.browser, ["com.apple.safari", "com.google.chrome", "org.mozilla.firefox", "com.microsoft.edgemac", "company.thebrowser.browser", "com.brave.browser", "com.operasoftware.opera", "com.vivaldi.vivaldi"]),
            (.development, ["com.apple.dt.xcode", "com.microsoft.vscode", "com.todesktop.230313mzl4w4u92", "dev.zed.zed", "com.jetbrains.", "com.sublimetext.", "com.panic.nova"]),
            (.terminal, ["com.apple.terminal", "com.googlecode.iterm2", "com.mitchellh.ghostty", "dev.warp.warp-stable", "org.alacritty", "net.kovidgoyal.kitty"]),
            (.mail, ["com.apple.mail", "com.microsoft.outlook", "com.readdle.smartemail-macos", "org.mozilla.thunderbird"]),
            (.messaging, ["com.apple.ichat", "com.apple.mobilesms", "com.tinyspeck.slackmacgap", "com.tencent.xinwechat", "com.tencent.qq", "com.hnc.discord", "ru.keepcoder.telegram", "org.whispersystems.signal-desktop", "com.microsoft.teams2", "com.openai.chat", "com.anthropic.claudefordesktop"]),
            (.writing, ["com.apple.notes", "com.apple.textedit", "com.apple.iwork.pages", "com.microsoft.word", "md.obsidian", "notion.id", "net.shinyfrog.bear", "abnerworks.typora"]),
            (.spreadsheet, ["com.apple.iwork.numbers", "com.microsoft.excel"]),
            (.creative, ["com.figma.desktop", "com.bohemiancoding.sketch3", "com.adobe.photoshop", "com.adobe.illustrator", "com.seriflabs.", "org.blenderfoundation.blender"]),
            (.fileManagement, ["com.apple.finder", "com.binarynights.forklift", "com.cocoatech.pathfinder"]),
        ]
        for (category, identifiers) in groups {
            if identifiers.contains(where: { $0.hasSuffix(".") ? id.hasPrefix($0) : id == $0 || id.hasPrefix($0 + ".") }) {
                return category
            }
        }
        return .unknown
    }
}

/// An explicit allowlist for inference. App identity and raw window titles stay in AppContext.
struct ModelContext: Codable, Sendable {
    var applicationCategory: ApplicationCategory
    var inputSurface: String
    var fieldRole: String
    var fieldLabel: String
    var selectedText: String
    var surroundingText: String
    var hasAccessibility: Bool
    var isSecure: Bool
}

extension AppContext {
    var modelContext: ModelContext {
        // Remove host branding only from AX metadata, never from the user's actual text.
        let hostPattern = "(?<![\\p{L}\\p{N}])" + NSRegularExpression.escapedPattern(for: appName) + "(?![\\p{L}\\p{N}])"
        let label = (appName.isEmpty ? fieldLabel : fieldLabel.replacingOccurrences(
            of: hostPattern, with: "", options: [.caseInsensitive, .regularExpression]
        )).trimmingCharacters(in: .whitespacesAndNewlines)
        let surface: String
        let patterns: [(String, String)] = [
            ("recipient", "^(?:to|cc|bcc|recipient|recipients|收件人|抄送|密送)(?:\\b|$)"),
            ("address_bar", "address bar|location bar|地址栏|网址栏"),
            ("search", "^(?:search|find|搜索|查找)(?:\\b|$)"),
            ("shell_prompt", "shell prompt|terminal prompt|终端命令"),
            ("code_editor", "code editor|source editor|代码编辑"),
            ("chat_composer", "message composer|chat input|聊天输入|消息输入"),
            ("color", "^(?:hex color|color value|颜色值|色值)$"),
            ("file_path", "^(?:file path|folder path|文件路径|目录路径)$"),
        ]
        if let match = patterns.first(where: { label.range(of: $0.1, options: [.regularExpression, .caseInsensitive]) != nil }) {
            surface = match.0
        } else {
            surface = ["AXTextField", "AXTextArea", "AXComboBox"].contains(fieldRole) ? "text" : "unknown"
        }
        return ModelContext(applicationCategory: ApplicationCategory.classify(bundleID: bundleID),
                            inputSurface: surface, fieldRole: fieldRole, fieldLabel: isSecure ? "" : label,
                            selectedText: isSecure ? "" : selectedText,
                            surroundingText: isSecure ? "" : surroundingText,
                            hasAccessibility: hasAccessibility, isSecure: isSecure)
    }
}
