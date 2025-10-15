
import AsyncStorage from "@react-native-async-storage/async-storage";
const KEY = "@goats:v1";
export async function loadGoats() { try { const json = await AsyncStorage.getItem(KEY); return json ? JSON.parse(json) : []; } catch { return []; } }
export async function saveGoats(data) { try { await AsyncStorage.setItem(KEY, JSON.stringify(data)); } catch {} }
