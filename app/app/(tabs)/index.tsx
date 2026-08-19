import { useEffect, useState } from "react";
import {
  Image,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";
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
import { LinearGradient } from "expo-linear-gradient";
import { OptionRow } from "../../src/components/OptionRow";
import { Logo } from "../../src/components/Logo";
import { ApiError, generateRoute } from "../../src/lib/api";
import { GRADE_OPTIONS, ANGLE_OPTIONS } from "../../src/lib/config";
import { colors, spacing } from "../../src/lib/theme";
import { useAuth } from "../../src/lib/auth";
import { saveRoute } from "../../src/lib/routes";

export default function GenerateScreen() {
  const [grade, setGrade] = useState(7);
  const [angle, setAngle] = useState(40);
  const [image, setImage] = useState<string | null>(null);
  const [imageBytes, setImageBytes] = useState<ArrayBuffer | null>(null);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { user, configured } = useAuth();

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
    <ScrollView style={styles.screen} contentContainerStyle={styles.content}>
      <View style={styles.header}>
        <Logo size={36} />
        <Text style={styles.title}>setter</Text>
      </View>
      <Text style={styles.subtitle}>Generate a Kilterboard route</Text>

      <View style={styles.controls}>
        <OptionRow label="Grade" options={GRADE_OPTIONS} value={grade} onChange={setGrade} formatOption={(v) => `V${v}`} />
        <OptionRow label="Angle" options={ANGLE_OPTIONS} value={angle} onChange={setAngle} formatOption={(v) => `${v}°`} />
      </View>

      <GenerateButton onPress={handleGenerate} loading={loading} />

      <View style={styles.imageArea}>
        {loading ? (
          <ShimmerPlaceholder />
        ) : image ? (
          <>
            <Animated.Image
              key={image}
              entering={FadeIn.duration(350)}
              source={{ uri: image }}
              style={styles.image}
              resizeMode="contain"
            />
            {configured && user ? (
              <Pressable style={styles.saveButton} onPress={handleSave} disabled={saving || saved}>
                <Text style={styles.saveButtonText}>
                  {saved ? "Saved" : saving ? "Saving…" : "Save route"}
                </Text>
              </Pressable>
            ) : null}
          </>
        ) : error ? (
          <Text style={styles.errorText}>{error}</Text>
        ) : (
          <Text style={styles.placeholderText}>Pick a grade and angle, then generate a climb.</Text>
        )}
      </View>
    </ScrollView>
  );
}

function GenerateButton({ onPress, loading }: { onPress: () => void; loading: boolean }) {
  const scale = useSharedValue(1);
  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ scale: scale.value }],
  }));

  return (
    <Pressable
      onPress={onPress}
      disabled={loading}
      onPressIn={() => {
        scale.value = withSpring(0.96, { damping: 14, stiffness: 300 });
      }}
      onPressOut={() => {
        scale.value = withSpring(1, { damping: 14, stiffness: 300 });
      }}
    >
      <Animated.View style={animatedStyle}>
        <LinearGradient
          colors={colors.accentGradient}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={styles.generateButton}
        >
          <Text style={styles.generateButtonText}>{loading ? "Generating…" : "Generate"}</Text>
        </LinearGradient>
      </Animated.View>
    </Pressable>
  );
}

function ShimmerPlaceholder() {
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

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.bg },
  content: { padding: spacing(3), alignItems: "stretch", gap: spacing(3), maxWidth: 560, width: "100%", alignSelf: "center" },
  header: { flexDirection: "row", alignItems: "center", justifyContent: "center", gap: spacing(1.25) },
  title: { color: colors.text, fontSize: 40, fontWeight: "800", textAlign: "center" },
  subtitle: { color: colors.textMuted, fontSize: 15, textAlign: "center", marginTop: -spacing(2) },
  controls: { gap: spacing(2.5) },
  generateButton: {
    borderRadius: 12,
    paddingVertical: spacing(1.75),
    alignItems: "center",
  },
  generateButtonText: { color: colors.accentText, fontSize: 17, fontWeight: "700" },
  imageArea: {
    minHeight: 380,
    borderRadius: 16,
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
    alignItems: "center",
    justifyContent: "center",
    padding: spacing(2),
    gap: spacing(2),
    overflow: "hidden",
  },
  shimmer: {
    width: "100%",
    height: "100%",
    minHeight: 340,
    borderRadius: 12,
    backgroundColor: colors.surfaceAlt,
  },
  image: { width: "100%", aspectRatio: 1 },
  placeholderText: { color: colors.textMuted, textAlign: "center" },
  errorText: { color: colors.bad, textAlign: "center" },
  saveButton: {
    borderWidth: 1,
    borderColor: colors.accent,
    borderRadius: 10,
    paddingVertical: spacing(1),
    paddingHorizontal: spacing(2.5),
  },
  saveButtonText: { color: colors.accent, fontWeight: "700" },
});
