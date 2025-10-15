
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import EnterGoatScreen from "../screens/EnterGoatScreen";
import BrowseGoatsScreen from "../screens/BrowseGoatsScreen";
import GoatDetailScreen from "../screens/GoatDetailScreen";
import BrowseAndDetailTablet from "../screens/BrowseAndDetailTablet";
import useDevice from "../hooks/useDevice";

const Tab = createBottomTabNavigator();
const Stack = createNativeStackNavigator();

function BrowseStack() {
  return (
    <Stack.Navigator>
      <Stack.Screen name="Browse" component={BrowseGoatsScreen} />
      <Stack.Screen name="GoatDetail" component={GoatDetailScreen} options={{ title: "Goat" }} />
    </Stack.Navigator>
  );
}

export default function RootNavigator() {
  const { isTablet } = useDevice();
  return (
    <Tab.Navigator screenOptions={{ headerShown: false }}>
      <Tab.Screen name="Enter" component={EnterGoatScreen} options={{ tabBarLabel: "Enter Goat" }} />
      {isTablet ? (
        <Tab.Screen name="List" component={BrowseAndDetailTablet} options={{ tabBarLabel: "Browse" }} />
      ) : (
        <Tab.Screen name="List" component={BrowseStack} options={{ tabBarLabel: "Browse" }} />
      )}
    </Tab.Navigator>
  );
}
