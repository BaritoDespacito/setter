import { Ionicons } from "@expo/vector-icons";
import { Pressable, View } from "react-native";
import { useTheme } from "../lib/theme-context";

interface StarRatingProps {
  value: number;
  onRate?: (stars: number) => void;
  size?: number;
}

export function StarRating({ value, onRate, size = 20 }: StarRatingProps) {
  const { colors } = useTheme();

  return (
    <View style={{ flexDirection: "row", gap: 2 }}>
      {[1, 2, 3, 4, 5].map((star) => (
        <Pressable key={star} onPress={onRate ? () => onRate(star) : undefined} disabled={!onRate}>
          <Ionicons
            name={star <= Math.round(value) ? "star" : "star-outline"}
            size={size}
            color={colors.accent}
          />
        </Pressable>
      ))}
    </View>
  );
}
