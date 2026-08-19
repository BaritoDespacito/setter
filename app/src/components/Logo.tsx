import Animated, { FadeIn } from "react-native-reanimated";
import Svg, { Circle, Line } from "react-native-svg";
import { useTheme } from "../lib/theme-context";

const AnimatedCircle = Animated.createAnimatedComponent(Circle);

// Same ascending-diagonal composition as the app icon (app/assets/icon.png) - 5
// circles growing toward the top-right, echoing both "connected route holds" and
// drawClimb()'s own circle-outline hold markers.
const POINTS: [number, number, number][] = [
  [14, 86, 7.5],
  [30, 68, 9],
  [48, 50, 10.5],
  [68, 32, 12],
  [86, 14, 14],
];

interface LogoProps {
  size?: number;
  animated?: boolean;
}

export function Logo({ size = 40, animated = true }: LogoProps) {
  const { colors } = useTheme();

  return (
    <Svg width={size} height={size} viewBox="0 0 100 100">
      {POINTS.slice(1).map((point, i) => {
        const [x1, y1, r1] = POINTS[i];
        const [x2, y2, r2] = point;
        const dx = x2 - x1;
        const dy = y2 - y1;
        const dist = Math.hypot(dx, dy) || 1;
        const ux = dx / dist;
        const uy = dy / dist;
        return (
          <Line
            key={`line-${i}`}
            x1={x1 + ux * r1}
            y1={y1 + uy * r1}
            x2={x2 - ux * r2}
            y2={y2 - uy * r2}
            stroke={colors.accent}
            strokeWidth={3.5}
            strokeLinecap="round"
          />
        );
      })}
      {POINTS.map(([x, y, r], i) => {
        const isLast = i === POINTS.length - 1;
        const circleProps = isLast
          ? { fill: colors.accent }
          : { fill: "transparent", stroke: colors.accent, strokeWidth: 3.5 };
        if (!animated) {
          return <Circle key={i} cx={x} cy={y} r={r} {...circleProps} />;
        }
        return (
          <AnimatedCircle
            key={i}
            cx={x}
            cy={y}
            r={r}
            {...circleProps}
            entering={FadeIn.delay(i * 80).duration(250)}
          />
        );
      })}
    </Svg>
  );
}
