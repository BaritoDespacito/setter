import { useMemo, useState } from "react";
import { Pressable, StyleSheet, Text, TextInput, View } from "react-native";
import { router } from "expo-router";
import { useAuth } from "../src/lib/auth";
import { fonts, spacing, type ThemeColors } from "../src/lib/theme";
import { useTheme } from "../src/lib/theme-context";

export default function LoginScreen() {
  const { colors } = useTheme();
  const styles = useMemo(() => makeStyles(colors), [colors]);
  const { signInWithPassword, signUpWithPassword } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [mode, setMode] = useState<"signin" | "signup">("signin");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async () => {
    setBusy(true);
    setError(null);
    const fn = mode === "signin" ? signInWithPassword : signUpWithPassword;
    const { error: err } = await fn(email.trim(), password);
    setBusy(false);
    if (err) {
      setError(err);
      return;
    }
    router.back();
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>{mode === "signin" ? "Sign in" : "Create account"}</Text>

      <TextInput
        style={styles.input}
        placeholder="Email"
        placeholderTextColor={colors.textMuted}
        autoCapitalize="none"
        keyboardType="email-address"
        value={email}
        onChangeText={setEmail}
      />
      <TextInput
        style={styles.input}
        placeholder="Password"
        placeholderTextColor={colors.textMuted}
        secureTextEntry
        value={password}
        onChangeText={setPassword}
      />

      {error ? <Text style={styles.error}>{error}</Text> : null}

      <Pressable style={styles.submit} onPress={submit} disabled={busy || !email || !password}>
        <Text style={styles.submitText}>
          {busy ? "Please wait…" : mode === "signin" ? "Sign in" : "Sign up"}
        </Text>
      </Pressable>

      <Pressable onPress={() => setMode(mode === "signin" ? "signup" : "signin")}>
        <Text style={styles.switchText}>
          {mode === "signin" ? "Need an account? Sign up" : "Already have an account? Sign in"}
        </Text>
      </Pressable>
    </View>
  );
}

function makeStyles(colors: ThemeColors) {
  return StyleSheet.create({
    container: { flex: 1, backgroundColor: colors.bg, padding: spacing(3), gap: spacing(2), justifyContent: "center" },
    title: { color: colors.text, fontFamily: fonts.display, fontSize: 24, marginBottom: spacing(1) },
    input: {
      backgroundColor: "transparent",
      borderBottomWidth: 1,
      borderBottomColor: colors.border,
      paddingVertical: spacing(1.25),
      fontFamily: fonts.body,
      color: colors.text,
    },
    error: { color: colors.bad, fontFamily: fonts.body },
    submit: { backgroundColor: colors.text, borderRadius: 4, padding: spacing(1.75), alignItems: "center", marginTop: spacing(1) },
    submitText: { color: colors.accentText, fontFamily: fonts.bodySemiBold, fontSize: 16 },
    switchText: { color: colors.accent, fontFamily: fonts.body, textAlign: "center", marginTop: spacing(1) },
  });
}
