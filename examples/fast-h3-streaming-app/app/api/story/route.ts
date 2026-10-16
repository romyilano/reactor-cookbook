import Anthropic from "@anthropic-ai/sdk";
import { NextRequest, NextResponse } from "next/server";
import {
  STORY_MAX_IMAGE_PAYLOAD,
  STORY_MAX_IMAGES,
  STORY_MODEL,
  type StoryHistoryItem,
  type StoryPlanRequest,
} from "@/lib/story-contract";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const VISION_DATA_URL = /^data:(image\/(?:jpeg|png));base64,([A-Za-z0-9+/=]+)$/;
const PLANNER_MODEL = process.env.STORY_MODEL?.trim() || STORY_MODEL;

function clientFrom() {
  const apiKey = process.env.ANTHROPIC_API_KEY?.trim();
  if (!apiKey) throw new Error("Missing ANTHROPIC_API_KEY.");
  return new Anthropic({ apiKey, maxRetries: 0 });
}

function historyFrom(value: unknown): StoryHistoryItem[] {
  if (!Array.isArray(value)) return [];
  return value.slice(-6).flatMap((item) => {
    if (!item || typeof item !== "object") return [];
    const record = item as Record<string, unknown>;
    return [{
      sceneSummary: typeof record.sceneSummary === "string" ? record.sceneSummary.slice(0, 1_200) : "",
      dialogue: typeof record.dialogue === "string" ? record.dialogue.slice(0, 500) : "",
      videoPrompt: typeof record.videoPrompt === "string" ? record.videoPrompt.slice(0, 2_000) : "",
    }];
  });
}

function imagesFrom(value: unknown) {
  if (!Array.isArray(value)) return [];
  const images = value.slice(0, STORY_MAX_IMAGES).filter((item): item is string => (
    typeof item === "string" && VISION_DATA_URL.test(item)
  ));
  if (images.reduce((sum, image) => sum + image.length, 0) > STORY_MAX_IMAGE_PAYLOAD) {
    throw new Error("Story planner image context is too large.");
  }
  return images.flatMap((image) => {
    const match = VISION_DATA_URL.exec(image);
    if (!match) return [];
    return [{
      type: "image" as const,
      source: { type: "base64" as const, media_type: match[1] as "image/jpeg" | "image/png", data: match[2] },
    }];
  });
}

function propsFrom(value: unknown) {
  if (!Array.isArray(value)) return [];
  return value.slice(0, 4).flatMap((item) => (
    typeof item === "string" && item.trim() ? [item.trim().slice(0, 500)] : []
  ));
}

const STORY_SCHEMA = {
  type: "object",
  properties: {
    videoPrompt: { type: "string" },
    sceneSummary: { type: "string" },
    dialogue: { type: "string" },
  },
  required: ["videoPrompt", "sceneSummary", "dialogue"],
  additionalProperties: false,
};

export function GET() {
  return NextResponse.json(
    { enabled: Boolean(process.env.ANTHROPIC_API_KEY?.trim()) },
    { headers: { "Cache-Control": "no-store, max-age=0" } },
  );
}

export async function POST(request: NextRequest) {
  try {
    const client = clientFrom();
    const body = (await request.json()) as Partial<StoryPlanRequest>;
    const direction = typeof body.direction === "string" ? body.direction.trim().slice(0, 4_000) : "";
    if (!direction) return NextResponse.json({ error: "A story direction is required." }, { status: 400 });
    const requestedDuration = typeof body.duration === "number" ? body.duration : 10;
    const duration = Math.min(14, Math.max(6, Math.round(requestedDuration)));

    const history = historyFrom(body.history);
    const images = imagesFrom(body.images);
    const props = propsFrom(body.props);
    const visualNote = images.length
      ? "Use the supplied image only as visual planning context."
      : "No reference image is available; rely on the written continuity history and textual scene cues.";
    const userText = [
      `Operator direction: ${direction}`,
      visualNote,
      props.length ? `Persistent prop requirements: ${props.join(" | ")}` : "No persistent prop is currently required.",
      `Recent accepted story history, oldest to newest: ${JSON.stringify(history)}`,
      `Plan only the next ${duration} seconds.`,
    ].join("\n\n");

    const startedAt = Date.now();
    const response = await client.messages.create({
      model: PLANNER_MODEL,
      max_tokens: 650,
      system: [
        "You are the live showrunner for an infinite audiovisual stream built from continuous short video clips.",
        "Continue the story causally from the newest history and visual frame while preserving character identity, location, lighting, screen direction, props, and camera language.",
        "Treat the operator direction as the requested next beat. Treat every named persistent prop as mandatory scene state and visibly incorporate it into the established scene in the most intuitive way.",
        "All newly added objects and images arrive as textual scene cues. Incorporate each named cue intuitively without cutting away from the established scene. Keep every named persistent prop visible or logically present across frames; subjects may receive, notice, pick up, prepare, move, or cook with them, but the props must not vanish.",
        `Write a concrete FastH3 generation prompt describing visible action over ${duration} seconds, camera behavior, ambience, sound effects, and a spoken line when dialogue helps the story.`,
        `Keep dialogue speakable within ${duration} seconds and quote the exact words inside videoPrompt. Do not mention prompts, source images, chunks, models, or continuity systems.`,
      ].join(" "),
      messages: [{ role: "user", content: [...images, { type: "text", text: userText }] }],
      output_config: { effort: "low", format: { type: "json_schema", schema: STORY_SCHEMA } },
    }, { timeout: 10_000 });

    if (response.stop_reason === "refusal") throw new Error("The story planner declined this request.");
    const textBlock = response.content.find((block) => block.type === "text");
    if (!textBlock || textBlock.type !== "text") throw new Error("The story planner returned no story plan.");
    const plan = JSON.parse(textBlock.text) as Record<string, unknown>;
    const videoPrompt = typeof plan.videoPrompt === "string" ? plan.videoPrompt.trim().slice(0, 8_000) : "";
    if (!videoPrompt) throw new Error("The story planner returned an empty video prompt.");

    return NextResponse.json({
      videoPrompt,
      sceneSummary: typeof plan.sceneSummary === "string" ? plan.sceneSummary.trim().slice(0, 2_000) : "",
      dialogue: typeof plan.dialogue === "string" ? plan.dialogue.trim().slice(0, 500) : "",
      model: PLANNER_MODEL,
      latencyMs: Date.now() - startedAt,
    });
  } catch (error) {
    if (error instanceof Anthropic.APIError) {
      return NextResponse.json({ error: error.message }, { status: error.status ?? 502 });
    }
    const message = error instanceof Error ? error.message : "Could not plan the next story beat.";
    const status = message.startsWith("Missing ANTHROPIC_API_KEY") ? 503 : 502;
    return NextResponse.json({ error: message }, { status });
  }
}
