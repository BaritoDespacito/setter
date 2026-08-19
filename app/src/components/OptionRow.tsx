import { Pressable, ScrollView, StyleSheet, Text, View } from "react-native";
import Animated, {
  useAnimatedStyle,
  withSpring,
  withTiming,
} from "react-native-reanimated";
import { colors, spacing } from "../lib/theme";

interface OptionRowProps {
  label: string;
  options: number[];
  value: number;
  onChange: (value: number) => void;
  formatOption?: (value: number) => string;
}

export function OptionRow({ label, options, value, onChange, formatOption }: OptionRowProps) {
  return (
    <View style={styles.container}>
      <Text style={styles.label}>{label}</Text>
      <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.chips}>
        {options.map((option) => (
          <Chip
            key={option}
            selected={option === value}
            label={formatOption ? formatOption(option) : String(option)}
            onPress={() => onChange(option)}
          />
        ))}
      </ScrollView>
    </View>
  );
}

function Chip({ selected, label, onPress }: { selected: boolean; label: string; onPress: () => void }) {
  const animatedStyle = useAnimatedStyle(() => ({
    backgroundColor: withTiming(selected ? colors.accent : colors.surfaceAlt, { duration: 180 }),
    borderColor: withTiming(selected ? colors.accent : colors.border, { duration: 180 }),
    transform: [{ scale: withSpring(selected ? 1.06 : 1, { damping: 12, stiffness: 220 }) }],
  }));

  return (
    <Pressable onPress={onPress}>
      <Animated.View style={[styles.chip, animatedStyle]}>
        <Text style={[styles.chipText, selected && styles.chipTextSelected]}>{label}</Text>
      </Animated.View>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  container: { gap: spacing(1) },
  label: {
    color: colors.textMuted,
    fontSize: 13,
    fontWeight: "600",
    textTransform: "uppercase",
    letterSpacing: 0.5,
  },
  chips: { gap: spacing(1), paddingVertical: spacing(0.5) },
  chip: {
    paddingHorizontal: spacing(2),
    paddingVertical: spacing(1),
    borderRadius: 999,
    backgroundColor: colors.surfaceAlt,
    borderWidth: 1,
    borderColor: colors.border,
  },
  chipText: { color: colors.text, fontWeight: "600" },
  chipTextSelected: { color: colors.accentText },
});
