import { useEffect, useState } from "react";
import { ActivityIndicator, ScrollView, StyleSheet, Text, View } from "react-native";
import { ChangelogEntry, fetchChangelog } from "../../src/lib/api";
import { colors, spacing } from "../../src/lib/theme";

function formatDate(iso: string) {
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" }) +
    " " + d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}

function rateColor(rate: number) {
  if (rate >= 0.75) return colors.good;
  if (rate >= 0.5) return colors.warn;
  return colors.bad;
}

function EntryCard({ entry }: { entry: ChangelogEntry }) {
  return (
    <View style={styles.card}>
      <View style={styles.cardHeader}>
        <Text style={styles.cardDate}>{formatDate(entry.timestamp)}</Text>
        <View style={[styles.badge, { backgroundColor: rateColor(entry.valid_rate) }]}>
          <Text style={styles.badgeText}>{Math.round(entry.valid_rate * 100)}% valid</Text>
        </View>
      </View>
      <Text style={styles.cardLabel}>{entry.label}</Text>
      <View style={styles.statsRow}>
        <Stat label="Routes" value={String(entry.total)} />
        <Stat label="Avg holds" value={entry.avg_holds.toFixed(1)} />
        <Stat label="Avg span" value={`${Math.round(entry.avg_y_span)}px`} />
        {entry.avg_critic_diff !== undefined ? (
          <Stat label="Critic err" value={`${entry.avg_critic_diff.toFixed(2)} gr`} />
        ) : null}
        {entry.val_loss !== null ? <Stat label="Val loss" value={entry.val_loss.toFixed(3)} /> : null}
      </View>
    </View>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.stat}>
      <Text style={styles.statValue}>{value}</Text>
      <Text style={styles.statLabel}>{label}</Text>
    </View>
  );
}

export default function ChangelogScreen() {
  const [entries, setEntries] = useState<ChangelogEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchChangelog()
      .then(setEntries)
      .catch(() => setError("Couldn't load the changelog right now."));
  }, []);

  return (
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
        entries.map((entry, i) => <EntryCard key={entry.timestamp + i} entry={entry} />)
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.bg },
  content: { padding: spacing(3), gap: spacing(2), maxWidth: 640, width: "100%", alignSelf: "center" },
  title: { color: colors.text, fontSize: 28, fontWeight: "800" },
  subtitle: { color: colors.textMuted, fontSize: 14, marginBottom: spacing(1) },
  errorText: { color: colors.textMuted, textAlign: "center", marginTop: spacing(4) },
  card: {
    backgroundColor: colors.surface,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing(2.5),
    gap: spacing(1),
  },
  cardHeader: { flexDirection: "row", justifyContent: "space-between", alignItems: "center" },
  cardDate: { color: colors.textMuted, fontSize: 12 },
  badge: { paddingHorizontal: spacing(1.25), paddingVertical: spacing(0.5), borderRadius: 999 },
  badgeText: { color: colors.accentText, fontSize: 12, fontWeight: "700" },
  cardLabel: { color: colors.text, fontSize: 15, fontWeight: "600" },
  statsRow: { flexDirection: "row", flexWrap: "wrap", gap: spacing(2.5), marginTop: spacing(0.5) },
  stat: { minWidth: 70 },
  statValue: { color: colors.text, fontSize: 16, fontWeight: "700" },
  statLabel: { color: colors.textMuted, fontSize: 11, textTransform: "uppercase" },
});
