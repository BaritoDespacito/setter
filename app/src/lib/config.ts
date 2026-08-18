export const API_URL =
  process.env.EXPO_PUBLIC_API_URL ?? "https://setter-api-490491172314.us-central1.run.app";

export const SUPABASE_URL = process.env.EXPO_PUBLIC_SUPABASE_URL ?? "";
export const SUPABASE_ANON_KEY = process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY ?? "";

export const MIN_GRADE = 1;
export const MAX_GRADE = 17;
export const MIN_ANGLE = 0;
export const MAX_ANGLE = 70;
export const ANGLE_STEP = 5;

export const GRADE_OPTIONS = Array.from(
  { length: MAX_GRADE - MIN_GRADE + 1 },
  (_, i) => MIN_GRADE + i
);

export const ANGLE_OPTIONS = Array.from(
  { length: (MAX_ANGLE - MIN_ANGLE) / ANGLE_STEP + 1 },
  (_, i) => MIN_ANGLE + i * ANGLE_STEP
);
