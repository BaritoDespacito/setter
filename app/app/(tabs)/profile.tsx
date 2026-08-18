import { router } from "expo-router";
import { Pressable, StyleSheet, Text, View } from "react-native";
import { useAuth } from "../../src/lib/auth";
import { colors, spacing } from "../../src/lib/theme";

export default function ProfileScreen() {
  const { user, configured, signOut } = useAuth();

  if (!configured) {
    return (
      <View style={styles.container}>
        <Text style={styles.title}>Accounts</Text>
        <Text style={styles.body}>Accounts aren't set up on this deployment yet.</Text>
      </View>
    );
  }

  if (!user) {
    return (
      <View style={styles.container}>
        <Text style={styles.title}>You're not signed in</Text>
        <Pressable style={styles.button} onPress={() => router.push("/login")}>
          <Text style={styles.buttonText}>Sign in / Sign up</Text>
        </Pressable>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Signed in</Text>
      <Text style={styles.body}>{user.email}</Text>
      <Pressable style={[styles.button, styles.buttonOutline]} onPress={() => signOut()}>
        <Text style={styles.buttonOutlineText}>Sign out</Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg, padding: spacing(3), gap: spacing(2), justifyContent: "center", alignItems: "center" },
  title: { color: colors.text, fontSize: 22, fontWeight: "800" },
  body: { color: colors.textMuted, fontSize: 15 },
  button: { backgroundColor: colors.accent, borderRadius: 10, paddingVertical: spacing(1.5), paddingHorizontal: spacing(3) },
  buttonText: { color: colors.accentText, fontWeight: "700" },
  buttonOutline: { backgroundColor: "transparent", borderWidth: 1, borderColor: colors.border },
  buttonOutlineText: { color: colors.text, fontWeight: "700" },
});
