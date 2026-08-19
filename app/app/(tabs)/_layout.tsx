import { Ionicons } from "@expo/vector-icons";
import { Tabs } from "expo-router";
import type { ColorValue } from "react-native";
import Animated, { useAnimatedStyle, withSpring } from "react-native-reanimated";
import { colors } from "../../src/lib/theme";

type IoniconName = keyof typeof Ionicons.glyphMap;

function TabIcon({ name, color, size, focused }: { name: IoniconName; color: ColorValue; size: number; focused: boolean }) {
  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ scale: withSpring(focused ? 1.15 : 1, { damping: 12, stiffness: 260 }) }],
  }));
  return (
    <Animated.View style={animatedStyle}>
      <Ionicons name={name} size={size} color={color} />
    </Animated.View>
  );
}

export default function TabsLayout() {
  return (
    <Tabs
      screenOptions={{
        headerStyle: { backgroundColor: colors.bg },
        headerTintColor: colors.text,
        headerShadowVisible: false,
        tabBarStyle: { backgroundColor: colors.surface, borderTopColor: colors.border },
        tabBarActiveTintColor: colors.accent,
        tabBarInactiveTintColor: colors.textMuted,
      }}
    >
      <Tabs.Screen
        name="index"
        options={{
          title: "Generate",
          tabBarIcon: ({ color, size, focused }) => <TabIcon name="shapes" size={size} color={color} focused={focused} />,
        }}
      />
      <Tabs.Screen
        name="saved"
        options={{
          title: "Saved",
          tabBarIcon: ({ color, size, focused }) => <TabIcon name="bookmark" size={size} color={color} focused={focused} />,
        }}
      />
      <Tabs.Screen
        name="changelog"
        options={{
          title: "Changelog",
          tabBarIcon: ({ color, size, focused }) => <TabIcon name="git-branch" size={size} color={color} focused={focused} />,
        }}
      />
      <Tabs.Screen
        name="profile"
        options={{
          title: "Profile",
          tabBarIcon: ({ color, size, focused }) => <TabIcon name="person-circle" size={size} color={color} focused={focused} />,
        }}
      />
    </Tabs>
  );
}
