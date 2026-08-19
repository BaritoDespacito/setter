import { useMemo } from "react";
import { router, usePathname } from "expo-router";
import { Pressable, StyleSheet, Text, View } from "react-native";
import Animated, { useAnimatedStyle, withTiming } from "react-native-reanimated";
import { fonts, spacing, type ThemeColors } from "../lib/theme";
import { useTheme } from "../lib/theme-context";

const DRAWER_WIDTH = 260;

const DESTINATIONS = [
  { href: "/", label: "Generate" },
  { href: "/saved", label: "Saved" },
  { href: "/changelog", label: "Changelog" },
  { href: "/profile", label: "Profile" },
] as const;

interface NavDrawerProps {
  open: boolean;
  onClose: () => void;
}

export function NavDrawer({ open, onClose }: NavDrawerProps) {
  const { colors } = useTheme();
  const pathname = usePathname();
  const styles = useMemo(() => makeStyles(colors), [colors]);

  const backdropStyle = useAnimatedStyle(() => ({
    opacity: withTiming(open ? 1 : 0, { duration: 200 }),
  }));
  const panelStyle = useAnimatedStyle(() => ({
    transform: [{ translateX: withTiming(open ? 0 : DRAWER_WIDTH, { duration: 220 }) }],
  }));

  return (
    <View style={StyleSheet.absoluteFill} pointerEvents={open ? "auto" : "none"}>
      <Animated.View style={[styles.backdrop, backdropStyle]}>
        <Pressable style={StyleSheet.absoluteFill} onPress={onClose} />
      </Animated.View>
      <Animated.View style={[styles.panel, panelStyle]}>
        {DESTINATIONS.map((dest) => {
          const active = dest.href === "/" ? pathname === "/" : pathname.startsWith(dest.href);
          return (
            <Pressable
              key={dest.href}
              onPress={() => {
                onClose();
                router.push(dest.href);
              }}
              style={styles.item}
            >
              <Text style={[styles.itemText, active && styles.itemTextActive]}>{dest.label}</Text>
            </Pressable>
          );
        })}
      </Animated.View>
    </View>
  );
}

function makeStyles(colors: ThemeColors) {
  return StyleSheet.create({
    backdrop: {
      position: "absolute",
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: "rgba(0,0,0,0.35)",
    },
    panel: {
      position: "absolute",
      top: 0,
      right: 0,
      bottom: 0,
      width: DRAWER_WIDTH,
      backgroundColor: colors.surface,
      borderLeftWidth: 1,
      borderLeftColor: colors.border,
      paddingTop: spacing(8),
      paddingHorizontal: spacing(3),
      gap: spacing(0.5),
    },
    item: { paddingVertical: spacing(1.5) },
    itemText: {
      fontFamily: fonts.displayMedium,
      fontSize: 20,
      color: colors.textMuted,
    },
    itemTextActive: {
      color: colors.text,
      fontFamily: fonts.display,
    },
  });
}
