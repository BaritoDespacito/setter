import { router } from "expo-router";
import { useCallback, useEffect, useMemo, useState } from "react";
import { ActivityIndicator, Image, Pressable, ScrollView, StyleSheet, Text, View } from "react-native";
import Animated, { FadeInDown } from "react-native-reanimated";
import { Masthead } from "../../src/components/Masthead";
import { NavDrawer } from "../../src/components/NavDrawer";
import { StarRating } from "../../src/components/StarRating";
import { useAuth } from "../../src/lib/auth";
import { deleteRoute, fetchSavedRoutes, rateRoute, SavedRoute } from "../../src/lib/routes";
import { fonts, spacing, type ThemeColors } from "../../src/lib/theme";
import { useTheme } from "../../src/lib/theme-context";

export default function SavedScreen() {
  const { colors } = useTheme();
  const styles = useMemo(() => makeStyles(colors), [colors]);
  const [drawerOpen, setDrawerOpen] = useState(false);
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

  const handleRate = (routeId: string, stars: number) => {
    if (!user) return;
    const previous = routes;
    setRoutes((rs) => rs?.map((r) => (r.id === routeId ? { ...r, my_stars: stars } : r)) ?? rs);
    rateRoute(routeId, user.id, stars).catch(() => {
      setRoutes(previous);
      setError("Couldn't save rating.");
    });
  };

  const handleDelete = (routeId: string) => {
    const previous = routes;
    setRoutes((rs) => rs?.filter((r) => r.id !== routeId) ?? rs);
    deleteRoute(routeId).catch(() => {
      setRoutes(previous);
      setError("Couldn't delete route.");
    });
  };

  const body = () => {
    if (!configured) {
      return <Text style={styles.msg}>Accounts aren't set up on this deployment yet.</Text>;
    }
    if (authLoading) {
      return <ActivityIndicator color={colors.accent} style={{ marginTop: spacing(4) }} />;
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
    if (error) return <Text style={styles.msg}>{error}</Text>;
    if (routes === null) return <ActivityIndicator color={colors.accent} style={{ marginTop: spacing(4) }} />;
    if (routes.length === 0) {
      return <Text style={styles.msg}>No saved routes yet — generate one and tap "Save route".</Text>;
    }
    return routes.map((r, i) => (
      <Animated.View key={r.id} style={styles.row} entering={FadeInDown.delay(i * 60).duration(300)}>
        <Image source={{ uri: r.image_url }} style={styles.image} resizeMode="contain" />
        <View style={styles.rowBody}>
          <Text style={styles.rowTitle}>V{r.grade} · {r.angle}°</Text>
          <StarRating value={r.my_stars ?? r.avg_stars ?? 0} onRate={(stars) => handleRate(r.id, stars)} />
          {r.rating_count > 0 ? (
            <Text style={styles.ratingSummary}>{r.avg_stars?.toFixed(1)} avg ({r.rating_count})</Text>
          ) : null}
          <Pressable onPress={() => handleDelete(r.id)}>
            <Text style={styles.deleteText}>Delete</Text>
          </Pressable>
        </View>
      </Animated.View>
    ));
  };

  return (
    <View style={styles.root}>
      <Masthead onMenuPress={() => setDrawerOpen(true)} />
      <ScrollView style={styles.screen} contentContainerStyle={styles.content}>
        <Text style={styles.title}>Saved routes</Text>
        {body()}
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
    centered: { alignItems: "center", gap: spacing(2), marginTop: spacing(4) },
    title: { color: colors.text, fontFamily: fonts.display, fontSize: 26, marginBottom: spacing(2) },
    msg: { color: colors.textMuted, fontFamily: fonts.body, textAlign: "center", marginTop: spacing(4) },
    button: { backgroundColor: colors.text, borderRadius: 4, paddingVertical: spacing(1.5), paddingHorizontal: spacing(3) },
    buttonText: { color: colors.accentText, fontFamily: fonts.bodySemiBold },
    row: {
      flexDirection: "row",
      borderBottomWidth: 1,
      borderBottomColor: colors.border,
      paddingVertical: spacing(2),
      gap: spacing(2),
    },
    image: { width: 96, height: 96, backgroundColor: colors.surfaceAlt },
    rowBody: { flex: 1, gap: spacing(0.75), justifyContent: "center" },
    rowTitle: { color: colors.text, fontFamily: fonts.bodySemiBold, fontSize: 16 },
    ratingSummary: { color: colors.textMuted, fontFamily: fonts.body, fontSize: 12 },
    deleteText: { color: colors.bad, fontFamily: fonts.body, fontSize: 13, marginTop: spacing(0.5) },
  });
}
