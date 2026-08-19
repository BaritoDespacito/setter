import { useEffect, useMemo, useState } from "react";
import { Pressable, ScrollView, StyleSheet, Text, View } from "react-native";
import Animated, {
  Easing,
  FadeIn,
  useAnimatedStyle,
  useSharedValue,
  withRepeat,
  withSequence,
  withSpring,
  withTiming,
} from "react-native-reanimated";
import { Masthead } from "../../src/components/Masthead";
import { NavDrawer } from "../../src/components/NavDrawer";
import { OptionRow } from "../../src/components/OptionRow";
import { ApiError, generateRoute } from "../../src/lib/api";
import { GRADE_OPTIONS, ANGLE_OPTIONS } from "../../src/lib/config";
import { fonts, spacing, type ThemeColors } from "../../src/lib/theme";
import { useTheme } from "../../src/lib/theme-context";
import { useAuth } from "../../src/lib/auth";
import { saveRoute } from "../../src/lib/routes";

export default function GenerateScreen() {
  const { colors } = useTheme();
  const styles = useMemo(() => makeStyles(colors), [colors]);
  const [drawerOpen, setDrawerOpen] = useState(false);

  const [grade, setGrade] = useState(7);
  const [angle, setAngle] = useState(40);
  const [image, setImage] = useState<string | null>(null);
  const [imageBytes, setImageBytes] = useState<ArrayBuffer | null>(null);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { user } = useAuth();

  const handleGenerate = async () => {
    setLoading(true);
    setError(null);
    setSaved(false);
    try {
      const { dataUri, bytes } = await generateRoute(grade, angle);
      setImage(dataUri);
      setImageBytes(bytes);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to generate a route. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    if (!user || !imageBytes) return;
    setSaving(true);
    try {
      const routeId = crypto.randomUUID();
      await saveRoute(user.id, routeId, grade, angle, imageBytes);
      setSaved(true);
    } catch {
      setError("Failed to save this route.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <View style={styles.root}>
      <Masthead onMenuPress={() => setDrawerOpen(true)} />
      <ScrollView style={styles.screen} contentContainerStyle={styles.content}>
        <Text style={styles.subtitle}>Generate a Kilterboard route</Text>

        <View style={styles.controls}>
          <OptionRow label="Grade" options={GRADE_OPTIONS} value={grade} onChange={setGrade} formatOption={(v) => `V${v}`} />
          <OptionRow label="Angle" options={ANGLE_OPTIONS} value={angle} onChange={setAngle} formatOption={(v) => `${v}°`} />
        </View>

        <GenerateButton onPress={handleGenerate} loading={loading} colors={colors} />

        <View style={styles.imageArea}>
          {loading ? (
            <ShimmerPlaceholder colors={colors} />
          ) : image ? (
            <>
              <Animated.Image
                key={image}
                entering={FadeIn.duration(350)}
                source={{ uri: image }}
                style={styles.image}
                resizeMode="contain"
              />
              <View style={styles.captionRow}>
                <Text style={styles.caption}>V{grade} · {angle}°</Text>
                {user ? (
                  <Pressable onPress={handleSave} disabled={saving || saved}>
                    <Text style={styles.saveLink}>{saved ? "Saved" : saving ? "Saving…" : "Save route"}</Text>
                  </Pressable>
                ) : null}
              </View>
            </>
          ) : error ? (
            <Text style={styles.errorText}>{error}</Text>
          ) : (
            <Text style={styles.placeholderText}>Pick a grade and angle, then generate a climb.</Text>
          )}
        </View>
      </ScrollView>
      <NavDrawer open={drawerOpen} onClose={() => setDrawerOpen(false)} />
    </View>
  );
}

function GenerateButton({ onPress, loading, colors }: { onPress: () => void; loading: boolean; colors: ThemeColors }) {
  const styles = useMemo(() => makeStyles(colors), [colors]);
  const scale = useSharedValue(1);
  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ scale: scale.value }],
  }));

  return (
    <Pressable
      onPress={onPress}
      disabled={loading}
      onPressIn={() => {
        scale.value = withSpring(0.97, { damping: 14, stiffness: 300 });
      }}
      onPressOut={() => {
        scale.value = withSpring(1, { damping: 14, stiffness: 300 });
      }}
    >
      <Animated.View style={[styles.generateButton, animatedStyle]}>
        <Text style={styles.generateButtonText}>{loading ? "Generating…" : "Generate"}</Text>
      </Animated.View>
    </Pressable>
  );
}

function ShimmerPlaceholder({ colors }: { colors: ThemeColors }) {
  const styles = useMemo(() => makeStyles(colors), [colors]);
  const opacity = useSharedValue(0.4);

  useEffect(() => {
    opacity.value = withRepeat(
      withSequence(
        withTiming(0.85, { duration: 700, easing: Easing.inOut(Easing.ease) }),
        withTiming(0.4, { duration: 700, easing: Easing.inOut(Easing.ease) })
      ),
      -1,
      false
    );
  }, [opacity]);

  const animatedStyle = useAnimatedStyle(() => ({ opacity: opacity.value }));

  return <Animated.View style={[styles.shimmer, animatedStyle]} />;
}

function makeStyles(colors: ThemeColors) {
  return StyleSheet.create({
    root: { flex: 1, backgroundColor: colors.bg },
    screen: { flex: 1 },
    content: { padding: spacing(3), gap: spacing(3), maxWidth: 640, width: "100%", alignSelf: "center" },
    subtitle: { color: colors.textMuted, fontFamily: fonts.body, fontSize: 15 },
    controls: { gap: spacing(3) },
    generateButton: {
      backgroundColor: colors.text,
      borderRadius: 4,
      paddingVertical: spacing(1.75),
      alignItems: "center",
    },
    generateButtonText: { color: colors.accentText, fontFamily: fonts.bodySemiBold, fontSize: 16 },
    imageArea: {
      borderTopWidth: 1,
      borderBottomWidth: 1,
      borderColor: colors.border,
      paddingVertical: spacing(3),
      minHeight: 320,
      alignItems: "center",
      justifyContent: "center",
      gap: spacing(1.5),
    },
    shimmer: {
      width: "100%",
      aspectRatio: 1,
      backgroundColor: colors.surfaceAlt,
    },
    image: { width: "100%", aspectRatio: 1 },
    captionRow: {
      flexDirection: "row",
      justifyContent: "space-between",
      width: "100%",
    },
    caption: { color: colors.textMuted, fontFamily: fonts.body, fontSize: 13 },
    saveLink: { color: colors.accent, fontFamily: fonts.bodySemiBold, fontSize: 13 },
    placeholderText: { color: colors.textMuted, fontFamily: fonts.body, textAlign: "center" },
    errorText: { color: colors.bad, fontFamily: fonts.body, textAlign: "center" },
  });
}
