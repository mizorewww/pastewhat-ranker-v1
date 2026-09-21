import Foundation

enum ClipKind: String, Codable, CaseIterable, Sendable {
    case text, url, email, code, command, phone, file, image, color

    var label: String {
        switch self {
        case .text: "文本"
        case .url: "链接"
        case .email: "邮箱"
        case .code: "代码"
        case .command: "命令"
        case .phone: "电话"
        case .file: "文件"
        case .image: "图片"
        case .color: "颜色"
        }
    }

    var symbol: String {
        switch self {
        case .text: "text.alignleft"
        case .url: "link"
        case .email: "envelope"
        case .code: "chevron.left.forwardslash.chevron.right"
        case .command: "terminal"
        case .phone: "phone"
        case .file: "doc"
        case .image: "photo"
        case .color: "paintpalette"
        }
    }
}

struct PasteboardPayload: Codable, Sendable {
    var representations: [String: Data]
}

struct ClipboardEntry: Codable, Identifiable, Sendable {
    var id: UUID = UUID()
    var copiedAt: Date = Date()
    var text: String
    var kind: ClipKind
    var sourceApp: String
    var sourceBundleID: String?
    var payloads: [PasteboardPayload] = []
    var fingerprint: String = ""

    var title: String {
        let line = text.split(whereSeparator: \.isNewline).first.map(String.init) ?? ""
        return line.isEmpty ? kind.label : String(line.prefix(180))
    }

    var subtitle: String {
        let lines = text.split(whereSeparator: \.isNewline)
        return lines.count > 1 ? String(lines.dropFirst().joined(separator: " ").prefix(180)) : ""
    }

    var searchText: String { "\(text) \(sourceApp) \(kind.label)" }

    var candidate: RecommendationCandidate {
        var value = rankerCandidate
        let text = value.text
        let excerpt = text.count > 2400 ? String(text.prefix(1600)) + "\n…\n" + String(text.suffix(800)) : text
        value.text = excerpt
        return value
    }

    var rankerCandidate: RecommendationCandidate {
        // The dedicated model shares one tokenizer-based preprocessing contract
        // with its teacher. Do not insert the older Laya head/tail excerpt here.
        let projection = CandidateProjection.project(payloads)
        return RecommendationCandidate(id: id.uuidString, text: projection.text, kind: projection.kind.rawValue,
                                       capabilities: projection.capabilities,
                                       sourceCategory: ApplicationCategory.classify(bundleID: sourceBundleID ?? ""))
    }
}

struct AppContext: Codable, Sendable {
    var appName: String = "当前应用"
    var bundleID: String = ""
    var processID: Int32 = 0
    var windowTitle: String = ""
    var fieldRole: String = ""
    var fieldLabel: String = ""
    var selectedText: String = ""
    var surroundingText: String = ""
    var hasAccessibility: Bool = false
    var isSecure: Bool = false

    var hasFieldContext: Bool {
        !isSecure && (!fieldLabel.isEmpty || !selectedText.isEmpty || !surroundingText.isEmpty)
    }

    var detail: String {
        if isSecure { return "安全输入框 · 不读取输入内容" }
        if !fieldLabel.isEmpty { return String(fieldLabel.prefix(90)) }
        if hasFieldContext { return "已读取当前输入位置的语境" }
        return hasAccessibility ? "当前应用语境" : "开启辅助功能可按输入语境推荐"
    }
}

struct RecommendationCandidate: Codable, Sendable {
    var id: String
    var text: String
    var kind: String
    var capabilities: [String]
    var sourceCategory: ApplicationCategory
}

struct RecommendationRequest: Codable, Sendable {
    var id: String
    var context: ModelContext
    var entries: [RecommendationCandidate]
}

struct RankedCandidate: Codable, Sendable {
    var id: String
    var score: Double
    var reason: String
}

struct RecommendationResponse: Codable, Sendable {
    var id: String
    var recommendedID: String?
    var rankings: [RankedCandidate]
    var mode: String
    var backend: String
    var elapsedMS: Double
    var message: String?
    var decision: String
    var shortlistedIDs: [String]
    var inferenceCount: Int
    var appliedFacets: [String]
    var modelVersion: String?

    var engineLabel: String {
        switch mode {
        case "jev": "JEV · 云端"
        case "ranker": "RANKER · 本机"
        case "laya": "LAYA · 本机"
        default: "本地匹配"
        }
    }

    var recommendationTitle: String {
        switch mode {
        case "jev": "Jev 为此刻推荐"
        case "ranker": "PasteWhat Ranker 为此刻推荐"
        case "laya": "Laya 为此刻推荐"
        default: "根据当前语境匹配"
        }
    }

    var statusText: String {
        if let message, !message.isEmpty { return message }
        guard recommendedID != nil else { return "没有明确推荐 · 按复制时间排列" }
        return mode == "laya" ? "Laya · \(backend.uppercased()) · 已在本机推荐" : "本地匹配 · 已找到相关内容"
    }
}

enum AppPaths {
    static var support: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("PasteWhat", isDirectory: true)
    }
}
