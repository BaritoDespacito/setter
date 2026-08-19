import { useMemo } from "react";
import { Ionicons } from "@expo/vector-icons";
import { Pressable, StyleSheet, Text, View } from "react-native";
import { fonts, spacing, type ThemeColors } from "../lib/theme";
import { useTheme } from "../lib/theme-context";
import { Logo } from "./Logo";

interface MastheadProps {
  onMenuPress: () => void;
}

export function Masthead({ onMenuPress }: MastheadProps) {
  const { colors } = useTheme();
  const styles = useMemo(() => makeStyles(colors), [colors]);

  return (
    <View style={styles.container}>
      <View style={styles.row}>
        <View style={styles.brand}>
          <Logo size={22} animated={false} />
          <Text style={styles.wordmark}>setter</Text>
        </View>
        <Pressable onPress={onMenuPress} hitSlop={12}>
          <Ionicons name="menu-outline" size={26} color={colors.text} />
        </Pressable>
      </View>
    </View>
  );
}

function makeStyles(colors: ThemeColors) {
  return StyleSheet.create({
    container: {
      borderBottomWidth: 1,
      borderBottomColor: colors.border,
    },
    row: {
      flexDirection: "row",
      alignItems: "center",
      justifyContent: "space-between",
      paddingHorizontal: spacing(3),
      paddingVertical: spacing(2),
      maxWidth: 640,
      width: "100%",
      alignSelf: "center",
    },
    brand: { flexDirection: "row", alignItems: "center", gap: spacing(1) },
    wordmark: { fontFamily: fonts.display, fontSize: 19, color: colors.text },
  });
}
