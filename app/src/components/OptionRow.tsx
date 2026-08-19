import { useMemo, useRef } from "react";
import { LayoutChangeEvent, Platform, Pressable, ScrollView, StyleSheet, Text, View } from "react-native";
import Animated, { useAnimatedStyle, useSharedValue, withSpring } from "react-native-reanimated";
import { fonts, spacing, type ThemeColors } from "../lib/theme";
import { useTheme } from "../lib/theme-context";

interface OptionRowProps {
  label: string;
  options: number[];
  value: number;
  onChange: (value: number) => void;
  formatOption?: (value: number) => string;
}

export function OptionRow({ label, options, value, onChange, formatOption }: OptionRowProps) {
  const { colors } = useTheme();
  const styles = useMemo(() => makeStyles(colors), [colors]);
  const scrollRef = useRef<ScrollView>(null);

  // On web, a horizontal ScrollView only responds to a trackpad's horizontal swipe or
  // shift+wheel by default - a plain vertical mouse-wheel scroll (the most common
  // input) does nothing, making the row look unscrollable. Redirect vertical wheel
  // delta into horizontal scroll so it behaves like any other horizontal carousel.
  const handleWheel =
    Platform.OS === "web"
      ? (e: { deltaX: number; deltaY: number; preventDefault: () => void }) => {
          if (Math.abs(e.deltaY) <= Math.abs(e.deltaX)) return;
          // react-native-web-specific escape hatch to the underlying scrollable DOM
          // node - not part of ScrollView's cross-platform ref type.
          const node = (scrollRef.current as unknown as { getScrollableNode?: () => HTMLElement })
            ?.getScrollableNode?.();
          if (!node) return;
          e.preventDefault();
          node.scrollLeft += e.deltaY;
        }
      : undefined;

  const underlineX = useSharedValue(0);
  const underlineWidth = useSharedValue(0);
  const layouts = useRef<Record<number, { x: number; width: number }>>({});

  const applyUnderline = (animate: boolean) => {
    const layout = layouts.current[value];
    if (!layout) return;
    if (animate) {
      underlineX.value = withSpring(layout.x, { damping: 18, stiffness: 260 });
      underlineWidth.value = withSpring(layout.width, { damping: 18, stiffness: 260 });
    } else {
      underlineX.value = layout.x;
      underlineWidth.value = layout.width;
    }
  };

  const handleLayout = (option: number) => (e: LayoutChangeEvent) => {
    const { x, width } = e.nativeEvent.layout;
    const isFirstMeasurement = !layouts.current[option];
    layouts.current[option] = { x, width };
    if (option === value) applyUnderline(!isFirstMeasurement);
  };

  const underlineStyle = useAnimatedStyle(() => ({
    transform: [{ translateX: underlineX.value }],
    width: underlineWidth.value,
  }));

  return (
    <View style={styles.container}>
      <Text style={styles.label}>{label}</Text>
      <ScrollView
        ref={scrollRef}
        horizontal
        showsHorizontalScrollIndicator={false}
        // @ts-expect-error - web-only DOM event, not part of ScrollViewProps' RN types
        onWheel={handleWheel}
      >
        <View>
          <View style={styles.optionsRow}>
            {options.map((option) => {
              const selected = option === value;
              return (
                <Pressable
                  key={option}
                  onPress={() => {
                    onChange(option);
                    applyUnderline(true);
                  }}
                  onLayout={handleLayout(option)}
                  style={styles.option}
                >
                  <Text style={[styles.optionText, selected && styles.optionTextSelected]}>
                    {formatOption ? formatOption(option) : option}
                  </Text>
                </Pressable>
              );
            })}
          </View>
          <Animated.View style={[styles.underline, underlineStyle]} />
        </View>
      </ScrollView>
    </View>
  );
}

function makeStyles(colors: ThemeColors) {
  return StyleSheet.create({
    container: { gap: spacing(1) },
    label: {
      color: colors.textMuted,
      fontFamily: fonts.bodySemiBold,
      fontSize: 12,
      textTransform: "uppercase",
      letterSpacing: 1,
    },
    optionsRow: { flexDirection: "row" },
    option: {
      paddingHorizontal: spacing(1.5),
      paddingVertical: spacing(1),
    },
    optionText: {
      fontFamily: fonts.body,
      fontSize: 16,
      color: colors.textMuted,
    },
    optionTextSelected: {
      fontFamily: fonts.bodySemiBold,
      color: colors.text,
    },
    underline: {
      position: "absolute",
      bottom: 0,
      height: 2,
      backgroundColor: colors.accent,
    },
  });
}
