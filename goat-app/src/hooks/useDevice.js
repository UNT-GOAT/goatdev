
import { useMemo } from "react";
import { useWindowDimensions, Platform } from "react-native";

export default function useDevice() {
  const { width, height } = useWindowDimensions();
  const isTablet = useMemo(() => Math.min(width, height) >= 768, [width, height]);
  const isIOS = Platform.OS === "ios";
  return { isTablet, isIOS, width, height };
}
