export const lightColors = {
  bg: "#F7F3EC",
  surface: "#FFFFFF",
  surfaceAlt: "#EFE9DE",
  border: "#DEDACE",
  text: "#1A1A17",
  textMuted: "#6B675E",
  accent: "#1F7A70",
  accentText: "#F7F3EC",
  good: "#2F7A55",
  warn: "#9C6B18",
  bad: "#B23B2E",
};

export const darkColors = {
  bg: "#0E0E0C",
  surface: "#18170F",
  surfaceAlt: "#211F17",
  border: "#332F24",
  text: "#F2EEE3",
  textMuted: "#8F8A7C",
  accent: "#3FBBAE",
  accentText: "#0A1211",
  good: "#4CC38A",
  warn: "#E5B94E",
  bad: "#E0685C",
};

export type ThemeColors = typeof lightColors;

export const fonts = {
  display: "Fraunces_600SemiBold",
  displayMedium: "Fraunces_500Medium",
  displayItalic: "Fraunces_500Medium_Italic",
  body: "Inter_400Regular",
  bodyMedium: "Inter_500Medium",
  bodySemiBold: "Inter_600SemiBold",
  bodyBold: "Inter_700Bold",
};

export const spacing = (n: number) => n * 8;
