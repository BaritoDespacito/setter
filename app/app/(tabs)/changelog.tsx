import { useEffect, useMemo, useState } from "react";
import { ActivityIndicator, ScrollView, StyleSheet, Text, View } from "react-native";
import Animated, { FadeInDown } from "react-native-reanimated";
import { Masthead } from "../../src/components/Masthead";
import { NavDrawer } from "../../src/components/NavDrawer";
import { ChangelogEntry, fetchChangelog } from "../../src/lib/api";
import { fonts, spacing, type ThemeColors } from "../../src/lib/theme";
import { useTheme } from "../../src/lib/theme-context";

function formatDate(iso: string) {
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" }) +
    " " + d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}

function rateColor(rate: number, colors: ThemeColors) {
  if (rate >= 0.75) return colors.good;
  if (rate >= 0.5) return colors.warn;
  return colors.bad;
}

function EntryRow({ entry, index, colors }: { entry: ChangelogEntry; index: number; colors: ThemeColors }) {
  const styles = useMemo(() => makeStyles(colors), [colors]);
  return (
    <Animated.View style={styles.row} entering={FadeInDown.delay(index * 60).duration(300)}>
      <View style={styles.rowHeader}>
        <Text style={styles.date}>{formatDate(entry.timestamp)}</Text>
        <Text style={[styles.validRate, { color: rateColor(entry.valid_rate, colors) }]}>
          {Math.round(entry.valid_rate * 100)}% valid
        </Text>
      </View>
      <Text style={styles.label}>{entry.label}</Text>
      <View style={styles.statsRow}>
        <Stat label="Routes" value={String(entry.total)} colors={colors} />
        <Stat label="Avg holds" value={entry.avg_holds.toFixed(1)} colors={colors} />
        <Stat label="Avg span" value={`${Math.round(entry.avg_y_span)}px`} colors={colors} />
        {entry.avg_critic_diff !== undefined ? (
          <Stat label="Critic err" value={`${entry.avg_critic_diff.toFixed(2)} gr`} colors={colors} />
        ) : null}
        {entry.val_loss !== null ? <Stat label="Val loss" value={entry.val_loss.toFixed(3)} colors={colors} /> : null}
      </View>
    </Animated.View>
  );
}

function Stat({ label, value, colors }: { label: string; value: string; colors: ThemeColors }) {
  const styles = useMemo(() => makeStyles(colors), [colors]);
  return (
    <View style={styles.stat}>
      <Text style={styles.statValue}>{value}</Text>
      <Text style={styles.statLabel}>{label}</Text>
    </View>
  );
}

export default function ChangelogScreen() {
  const { colors } = useTheme();
  const styles = useMemo(() => makeStyles(colors), [colors]);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [entries, setEntries] = useState<ChangelogEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchChangelog()
      .then(setEntries)
      .catch(() => setError("Couldn't load the changelog right now."));
  }, []);

  return (
    <View style={styles.root}>
      <Masthead onMenuPress={() => setDrawerOpen(true)} />
      <ScrollView style={styles.screen} contentContainerStyle={styles.content}>
        <Text style={styles.title}>Changelog</Text>
        <Text style={styles.subtitle}>Model quality over time, tracked automatically after each training run.</Text>

        {error ? (
          <Text style={styles.errorText}>{error}</Text>
        ) : entries === null ? (
          <ActivityIndicator color={colors.accent} style={{ marginTop: spacing(4) }} />
        ) : entries.length === 0 ? (
          <Text style={styles.errorText}>No evaluation history yet.</Text>
        ) : (
          entries.map((entry, i) => (
            <EntryRow key={entry.timestamp + i} entry={entry} index={i} colors={colors} />
          ))
        )}
      </ScrollView>
      <NavDrawer open={drawerOpen} onClose={() => setDrawerOpen(false)} />
    </View>
  );
}

function makeStyles(colors: ThemeColors) {
  return StyleSheet.create({
    root: { flex: 1, backgroundColor: colors.bg },
    screen: { flex: 1 },
    content: { padding: spacing(3), maxWidth: 640, width: "100%", alignSelf: "center" },
    title: { color: colors.text, fontFamily: fonts.display, fontSize: 26, marginBottom: spacing(0.5) },
    subtitle: { color: colors.textMuted, fontFamily: fonts.body, fontSize: 14, marginBottom: spacing(3) },
    errorText: { color: colors.textMuted, fontFamily: fonts.body, textAlign: "center", marginTop: spacing(4) },
    row: {
      borderBottomWidth: 1,
      borderBottomColor: colors.border,
      paddingVertical: spacing(2.5),
      gap: spacing(0.75),
    },
    rowHeader: { flexDirection: "row", justifyContent: "space-between", alignItems: "center" },
    date: { color: colors.textMuted, fontFamily: fonts.body, fontSize: 12 },
    validRate: { fontFamily: fonts.bodySemiBold, fontSize: 12 },
    label: { color: colors.text, fontFamily: fonts.bodySemiBold, fontSize: 15 },
    statsRow: { flexDirection: "row", flexWrap: "wrap", gap: spacing(3), marginTop: spacing(0.5) },
    stat: { minWidth: 70 },
    statValue: { color: colors.text, fontFamily: fonts.display, fontSize: 17 },
    statLabel: { color: colors.textMuted, fontFamily: fonts.body, fontSize: 11, textTransform: "uppercase" },
  });
}
