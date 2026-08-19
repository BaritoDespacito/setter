import { router } from "expo-router";
import { useMemo, useState } from "react";
import { Pressable, ScrollView, StyleSheet, Text, View } from "react-native";
import { Masthead } from "../../src/components/Masthead";
import { NavDrawer } from "../../src/components/NavDrawer";
import { useAuth } from "../../src/lib/auth";
import { fonts, spacing, type ThemeColors } from "../../src/lib/theme";
import { useTheme } from "../../src/lib/theme-context";

const PREFERENCES = [
  { value: "system", label: "System" },
  { value: "light", label: "Light" },
  { value: "dark", label: "Dark" },
] as const;

function ThemeToggle({ colors }: { colors: ThemeColors }) {
  const { preference, setPreference } = useTheme();
  const styles = useMemo(() => makeStyles(colors), [colors]);

  return (
    <View style={styles.toggleRow}>
      {PREFERENCES.map((p) => {
        const active = p.value === preference;
        return (
          <Pressable key={p.value} onPress={() => setPreference(p.value)} style={styles.toggleItem}>
            <Text style={[styles.toggleText, active && styles.toggleTextActive]}>{p.label}</Text>
          </Pressable>
        );
      })}
    </View>
  );
}

export default function ProfileScreen() {
  const { colors } = useTheme();
  const styles = useMemo(() => makeStyles(colors), [colors]);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const { user, configured, signOut } = useAuth();

  return (
    <View style={styles.root}>
      <Masthead onMenuPress={() => setDrawerOpen(true)} />
      <ScrollView style={styles.screen} contentContainerStyle={styles.content}>
        {!configured ? (
          <Text style={styles.body}>Accounts aren't set up on this deployment yet.</Text>
        ) : !user ? (
          <View style={styles.section}>
            <Text style={styles.title}>You're not signed in</Text>
            <Pressable style={styles.button} onPress={() => router.push("/login")}>
              <Text style={styles.buttonText}>Sign in / Sign up</Text>
            </Pressable>
          </View>
        ) : (
          <View style={styles.section}>
            <Text style={styles.title}>Signed in</Text>
            <Text style={styles.body}>{user.email}</Text>
            <Pressable style={[styles.button, styles.buttonOutline]} onPress={() => signOut()}>
              <Text style={styles.buttonOutlineText}>Sign out</Text>
            </Pressable>
          </View>
        )}

        <View style={styles.section}>
          <Text style={styles.sectionLabel}>Appearance</Text>
          <ThemeToggle colors={colors} />
        </View>
      </ScrollView>
      <NavDrawer open={drawerOpen} onClose={() => setDrawerOpen(false)} />
    </View>
  );
}

function makeStyles(colors: ThemeColors) {
  return StyleSheet.create({
    root: { flex: 1, backgroundColor: colors.bg },
    screen: { flex: 1 },
    content: { padding: spacing(3), gap: spacing(4), maxWidth: 640, width: "100%", alignSelf: "center" },
    section: { gap: spacing(1.5) },
    sectionLabel: {
      color: colors.textMuted,
      fontFamily: fonts.bodySemiBold,
      fontSize: 12,
      textTransform: "uppercase",
      letterSpacing: 1,
    },
    title: { color: colors.text, fontFamily: fonts.display, fontSize: 22 },
    body: { color: colors.textMuted, fontFamily: fonts.body, fontSize: 15 },
    button: { backgroundColor: colors.text, borderRadius: 4, paddingVertical: spacing(1.5), paddingHorizontal: spacing(3), alignSelf: "flex-start" },
    buttonText: { color: colors.accentText, fontFamily: fonts.bodySemiBold },
    buttonOutline: { backgroundColor: "transparent", borderWidth: 1, borderColor: colors.border },
    buttonOutlineText: { color: colors.text, fontFamily: fonts.bodySemiBold },
    toggleRow: {
      flexDirection: "row",
      borderWidth: 1,
      borderColor: colors.border,
      borderRadius: 4,
      alignSelf: "flex-start",
      overflow: "hidden",
    },
    toggleItem: { paddingVertical: spacing(1), paddingHorizontal: spacing(2) },
    toggleText: { color: colors.textMuted, fontFamily: fonts.body, fontSize: 14 },
    toggleTextActive: { color: colors.text, fontFamily: fonts.bodySemiBold },
  });
}
