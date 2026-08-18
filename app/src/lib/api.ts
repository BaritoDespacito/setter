import { API_URL } from "./config";

export class ApiError extends Error {}

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  const chunkSize = 0x8000;
  let binary = "";
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
  }
  return btoa(binary);
}

export async function generateRoute(grade: number, angle: number): Promise<string> {
  const resp = await fetch(`${API_URL}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ grade, angle }),
  });

  if (!resp.ok) {
    let message = `Generation failed (${resp.status})`;
    try {
      const body = await resp.json();
      if (body?.error) message = body.error;
    } catch {
      // response wasn't JSON, keep default message
    }
    throw new ApiError(message);
  }

  const buffer = await resp.arrayBuffer();
  return `data:image/png;base64,${arrayBufferToBase64(buffer)}`;
}

export interface PerGradeStat {
  valid_rate: number;
  avg_holds: number;
  avg_y_span: number;
  avg_critic_diff?: number;
}

export interface ChangelogEntry {
  timestamp: string;
  label: string;
  checkpoint: string | null;
  val_loss: number | null;
  total: number;
  valid_rate: number;
  avg_holds: number;
  avg_y_span: number;
  avg_critic_diff?: number;
  per_grade: Record<string, PerGradeStat>;
}

export async function fetchChangelog(): Promise<ChangelogEntry[]> {
  const resp = await fetch(`${API_URL}/changelog`);
  if (!resp.ok) {
    throw new ApiError(`Failed to load changelog (${resp.status})`);
  }
  const data = await resp.json();
  return data.entries as ChangelogEntry[];
}
