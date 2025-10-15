
import "react-native-gesture-handler";
import { NavigationContainer } from "@react-navigation/native";
import { SafeAreaProvider } from "react-native-safe-area-context";
import RootNavigator from "./src/navigation/RootNavigator";
import { GoatProvider } from "./src/context/GoatContext";

export default function App() {
  return (
    <SafeAreaProvider>
      <GoatProvider>
        <NavigationContainer>
          <RootNavigator />
        </NavigationContainer>
      </GoatProvider>
    </SafeAreaProvider>
  );
}
