import { useState } from "react";
import { Pressable, StyleSheet, Text, TextInput, View } from "react-native";
import { router } from "expo-router";
import { useAuth } from "../src/lib/auth";
import { colors, spacing } from "../src/lib/theme";

export default function LoginScreen() {
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

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg, padding: spacing(3), gap: spacing(2), justifyContent: "center" },
  title: { color: colors.text, fontSize: 24, fontWeight: "800", marginBottom: spacing(1) },
  input: {
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: 10,
    padding: spacing(1.5),
    color: colors.text,
  },
  error: { color: colors.bad },
  submit: { backgroundColor: colors.accent, borderRadius: 10, padding: spacing(1.75), alignItems: "center", marginTop: spacing(1) },
  submitText: { color: colors.accentText, fontWeight: "700", fontSize: 16 },
  switchText: { color: colors.accent, textAlign: "center", marginTop: spacing(1) },
});
