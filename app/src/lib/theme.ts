export const colors = {
  bg: "#0B1110",
  surface: "#141D1C",
  surfaceAlt: "#1D2C29",
  border: "#2E3D3A",
  text: "#F5F7F6",
  textMuted: "#A3B6B2",
  accent: "#35B0A6",
  accentText: "#03130F",
  accent2: "#FF8A4C",
  accent2Text: "#1C0900",
  good: "#4CC38A",
  warn: "#E5B94E",
  bad: "#E0685C",
  accentGradient: ["#35B0A6", "#1E7A72"] as const,
};

export const spacing = (n: number) => n * 8;
