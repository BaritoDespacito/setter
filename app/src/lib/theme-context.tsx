import AsyncStorage from "@react-native-async-storage/async-storage";
import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { Platform, useColorScheme } from "react-native";
import { darkColors, lightColors, type ThemeColors } from "./theme";

type ThemePreference = "system" | "light" | "dark";
const STORAGE_KEY = "setter.themePreference";

interface ThemeContextValue {
  colors: ThemeColors;
  scheme: "light" | "dark";
  preference: ThemePreference;
  setPreference: (p: ThemePreference) => void;
}

const ThemeContext = createContext<ThemeContextValue | null>(null);

function isPreference(v: unknown): v is ThemePreference {
  return v === "light" || v === "dark" || v === "system";
}

// On web, read synchronously from localStorage directly rather than through
// AsyncStorage's web shim - avoids a render with the wrong theme before the async
// AsyncStorage.getItem() resolves (or, if it fails to resolve at all, getting stuck
// on the wrong theme permanently).
function readInitialPreference(): ThemePreference {
  if (Platform.OS === "web" && typeof window !== "undefined") {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (isPreference(stored)) return stored;
  }
  return "system";
}

// react-native-web's useColorScheme() can report an incorrect default (e.g. "dark")
// for the first render or two before its matchMedia listener settles, which - since
// nothing here waits on that - lets components downstream commit real DOM nodes
// (with inline styles baked from that wrong value) before the hook corrects itself.
// Reading matchMedia directly and synchronously on web sidesteps that window
// entirely rather than trying to patch up whatever rendered before the correction.
function readInitialSystemScheme(): "light" | "dark" {
  if (Platform.OS === "web" && typeof window !== "undefined" && window.matchMedia) {
    return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  }
  return "dark";
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const hookScheme = useColorScheme();
  const [webScheme, setWebScheme] = useState<"light" | "dark">(readInitialSystemScheme);

  useEffect(() => {
    if (Platform.OS !== "web" || typeof window === "undefined" || !window.matchMedia) return;
    const mq = window.matchMedia("(prefers-color-scheme: light)");
    const listener = (e: MediaQueryListEvent) => setWebScheme(e.matches ? "light" : "dark");
    mq.addEventListener("change", listener);
    return () => mq.removeEventListener("change", listener);
  }, []);

  const systemScheme = Platform.OS === "web" ? webScheme : hookScheme === "light" ? "light" : "dark";
  const [preference, setPreferenceState] = useState<ThemePreference>(readInitialPreference);

  useEffect(() => {
    if (Platform.OS === "web") return;
    AsyncStorage.getItem(STORAGE_KEY).then((stored) => {
      if (isPreference(stored)) setPreferenceState(stored);
    });
  }, []);

  const setPreference = (p: ThemePreference) => {
    setPreferenceState(p);
    if (Platform.OS === "web" && typeof window !== "undefined") {
      window.localStorage.setItem(STORAGE_KEY, p);
    } else {
      AsyncStorage.setItem(STORAGE_KEY, p);
    }
  };

  const scheme = preference === "system" ? systemScheme : preference;
  const colors = scheme === "dark" ? darkColors : lightColors;
  const value = useMemo(
    () => ({ colors, scheme, preference, setPreference }),
    [colors, scheme, preference]
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme() {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme must be used within ThemeProvider");
  return ctx;
}
