/** Bind one synthetic teacher completion to the user's existing Pi/Devin provider. */
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import devinExtension from "../../pi-devin/extensions/index.ts";
import { resolveModelUid } from "../../pi-devin/src/models.ts";

interface TeacherRequest {
  system: string;
  user: string;
  max_tokens: number;
  thinking: "medium" | "high" | "max";
}

export default async function teacherExtension(pi: ExtensionAPI): Promise<void> {
  const requestPath = process.env.PASTEWHAT_PI_REQUEST_FILE;
  const receiptPath = process.env.PASTEWHAT_PI_RECEIPT_FILE;
  if (!requestPath || !receiptPath) throw new Error("Missing bound teacher request/receipt paths");
  const bytes = readFileSync(requestPath);
  const request: TeacherRequest = JSON.parse(bytes.toString("utf8"));
  if (typeof request.system !== "string" || typeof request.user !== "string"
      || !Number.isInteger(request.max_tokens) || request.max_tokens < 1 || request.max_tokens > 131072
      || !["medium", "high", "max"].includes(request.thinking)) {
    throw new Error("Invalid bound teacher request");
  }
  const requestHash = createHash("sha256").update(bytes).digest("hex");
  let calls = 0;
  const delegated: ExtensionAPI = {
    ...pi,
    registerProvider(name, config) {
      if (typeof name !== "string" || name !== "devin" || !config?.streamSimple) {
        throw new Error("The teacher wrapper accepts only the existing Devin stream provider");
      }
      const original = config.streamSimple;
      pi.registerProvider(name, {
        ...config,
        streamSimple(model, _context, options) {
          calls += 1;
          if (calls !== 1) throw new Error("Only one provider completion is allowed per teacher process");
          if (model.provider !== "devin" || model.id !== "swe-2") {
            throw new Error("The bound teacher must be devin/swe-2");
          }
          const mapped = model.thinkingLevelMap?.[request.thinking];
          if (typeof mapped !== "string") throw new Error("Requested SWE-2 thinking level is unavailable");
          const actualModel = resolveModelUid(model.id, model.thinkingLevelMap, request.thinking);
          if (actualModel !== mapped) throw new Error("SWE-2 variant mapping changed");
          const receipt = {
            version: "pastewhat-pi-teacher-receipt-v1",
            transport: "pi-cli-json",
            provider: "devin",
            requested_model: "swe-2",
            actual_model: actualModel,
            request_sha256: requestHash,
            max_tokens: request.max_tokens,
            thinking: request.thinking,
            provider_call_count: calls,
            isolated: true,
            context_message_count: 1,
            tools_count: 0,
            usage_source: "pi-devin-provider-reported-or-unknown",
          };
          writeFileSync(receiptPath, JSON.stringify(receipt) + "\n", { mode: 0o600 });
          // The legacy streamSimple provider uses this exact Context shape.
          // Workspace, skills, Pi's own prompt and previous turns are excluded.
          const context = {
            systemPrompt: request.system,
            messages: [{ role: "user" as const, content: request.user, timestamp: 0 }],
            tools: [],
          };
          return original(model, context, { ...options, maxTokens: request.max_tokens, reasoning: request.thinking });
        },
      });
    },
  };
  await devinExtension(delegated);
}
