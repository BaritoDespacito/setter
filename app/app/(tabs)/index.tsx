import { useState } from "react";
import {
  ActivityIndicator,
  Image,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { OptionRow } from "../../src/components/OptionRow";
import { ApiError, generateRoute } from "../../src/lib/api";
import { GRADE_OPTIONS, ANGLE_OPTIONS } from "../../src/lib/config";
import { colors, spacing } from "../../src/lib/theme";
import { useAuth } from "../../src/lib/auth";
import { supabase } from "../../src/lib/supabase";

export default function GenerateScreen() {
  const [grade, setGrade] = useState(7);
  const [angle, setAngle] = useState(40);
  const [image, setImage] = useState<string | null>(null);
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
      const dataUri = await generateRoute(grade, angle);
      setImage(dataUri);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to generate a route. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    if (!supabase || !user || !image) return;
    setSaving(true);
    try {
      const { error: saveError } = await supabase.from("routes").insert({
        user_id: user.id,
        grade,
        angle,
        image_data_uri: image,
      });
      if (saveError) throw saveError;
      setSaved(true);
    } catch {
      setError("Failed to save this route.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <ScrollView style={styles.screen} contentContainerStyle={styles.content}>
      <Text style={styles.title}>setter</Text>
      <Text style={styles.subtitle}>Generate a Kilterboard route</Text>

      <View style={styles.controls}>
        <OptionRow label="Grade" options={GRADE_OPTIONS} value={grade} onChange={setGrade} formatOption={(v) => `V${v}`} />
        <OptionRow label="Angle" options={ANGLE_OPTIONS} value={angle} onChange={setAngle} formatOption={(v) => `${v}°`} />
      </View>

      <Pressable style={styles.generateButton} onPress={handleGenerate} disabled={loading}>
        <Text style={styles.generateButtonText}>{loading ? "Generating…" : "Generate"}</Text>
      </Pressable>

      <View style={styles.imageArea}>
        {loading ? (
          <ActivityIndicator size="large" color={colors.accent} />
        ) : image ? (
          <>
            <Image source={{ uri: image }} style={styles.image} resizeMode="contain" />
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

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.bg },
  content: { padding: spacing(3), alignItems: "stretch", gap: spacing(3), maxWidth: 560, width: "100%", alignSelf: "center" },
  title: { color: colors.text, fontSize: 40, fontWeight: "800", textAlign: "center" },
  subtitle: { color: colors.textMuted, fontSize: 15, textAlign: "center", marginTop: -spacing(2) },
  controls: { gap: spacing(2.5) },
  generateButton: {
    backgroundColor: colors.accent,
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
