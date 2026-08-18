import { router } from "expo-router";
import { useCallback, useEffect, useState } from "react";
import { ActivityIndicator, Image, Pressable, ScrollView, StyleSheet, Text, View } from "react-native";
import { StarRating } from "../../src/components/StarRating";
import { useAuth } from "../../src/lib/auth";
import { deleteRoute, fetchSavedRoutes, rateRoute, SavedRoute } from "../../src/lib/routes";
import { colors, spacing } from "../../src/lib/theme";

export default function SavedScreen() {
  const { user, configured, loading: authLoading } = useAuth();
  const [routes, setRoutes] = useState<SavedRoute[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    if (!user) return;
    fetchSavedRoutes(user.id)
      .then(setRoutes)
      .catch(() => setError("Couldn't load your saved routes."));
  }, [user]);

  useEffect(() => {
    load();
  }, [load]);

  if (!configured) {
    return (
      <View style={styles.centered}>
        <Text style={styles.msg}>Accounts aren't set up on this deployment yet.</Text>
      </View>
    );
  }

  if (authLoading) {
    return (
      <View style={styles.centered}>
        <ActivityIndicator color={colors.accent} />
      </View>
    );
  }

  if (!user) {
    return (
      <View style={styles.centered}>
        <Text style={styles.msg}>Sign in to save and rate routes.</Text>
        <Pressable style={styles.button} onPress={() => router.push("/login")}>
          <Text style={styles.buttonText}>Sign in</Text>
        </Pressable>
      </View>
    );
  }

  return (
    <ScrollView style={styles.screen} contentContainerStyle={styles.content}>
      <Text style={styles.title}>Saved routes</Text>

      {error ? (
        <Text style={styles.msg}>{error}</Text>
      ) : routes === null ? (
        <ActivityIndicator color={colors.accent} style={{ marginTop: spacing(4) }} />
      ) : routes.length === 0 ? (
        <Text style={styles.msg}>No saved routes yet — generate one and tap "Save route".</Text>
      ) : (
        routes.map((r) => (
          <View key={r.id} style={styles.card}>
            <Image source={{ uri: r.image_data_uri }} style={styles.image} resizeMode="contain" />
            <View style={styles.cardBody}>
              <Text style={styles.cardTitle}>V{r.grade} · {r.angle}°</Text>
              <StarRating
                value={r.my_stars ?? r.avg_stars ?? 0}
                onRate={(stars) =>
                  rateRoute(r.id, user.id, stars).then(load).catch(() => setError("Couldn't save rating."))
                }
              />
              {r.rating_count > 0 ? (
                <Text style={styles.ratingSummary}>
                  {r.avg_stars?.toFixed(1)} avg ({r.rating_count})
                </Text>
              ) : null}
              <Pressable
                onPress={() => deleteRoute(r.id).then(load).catch(() => setError("Couldn't delete route."))}
              >
                <Text style={styles.deleteText}>Delete</Text>
              </Pressable>
            </View>
          </View>
        ))
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.bg },
  content: { padding: spacing(3), gap: spacing(2), maxWidth: 640, width: "100%", alignSelf: "center" },
  centered: { flex: 1, backgroundColor: colors.bg, alignItems: "center", justifyContent: "center", gap: spacing(2), padding: spacing(3) },
  title: { color: colors.text, fontSize: 28, fontWeight: "800", marginBottom: spacing(1) },
  msg: { color: colors.textMuted, textAlign: "center" },
  button: { backgroundColor: colors.accent, borderRadius: 10, paddingVertical: spacing(1.5), paddingHorizontal: spacing(3) },
  buttonText: { color: colors.accentText, fontWeight: "700" },
  card: {
    flexDirection: "row",
    backgroundColor: colors.surface,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: colors.border,
    overflow: "hidden",
  },
  image: { width: 120, height: 120, backgroundColor: colors.surfaceAlt },
  cardBody: { flex: 1, padding: spacing(2), gap: spacing(0.75), justifyContent: "center" },
  cardTitle: { color: colors.text, fontSize: 16, fontWeight: "700" },
  ratingSummary: { color: colors.textMuted, fontSize: 12 },
  deleteText: { color: colors.bad, fontSize: 13, marginTop: spacing(0.5) },
});
