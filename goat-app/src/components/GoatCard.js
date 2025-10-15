
import { View, Text, Pressable, StyleSheet } from "react-native";
export default function GoatCard({ goat, onPress }) {
  return (
    <Pressable onPress={onPress} style={styles.card}>
      <View style={{ flexDirection: "row", justifyContent: "space-between" }}>
        <Text style={styles.title}>#{goat.id}</Text>
        <Text style={styles.badge}>{goat.grade ?? "—"}</Text>
      </View>
      <Text>Weight: {goat.weight} kg</Text>
      <Text>Height: {goat.height} cm</Text>
      <Text>Length: {goat.length} cm</Text>
      {goat.notes ? <Text numberOfLines={2}>Notes: {goat.notes}</Text> : null}
    </Pressable>
  );
}
const styles = StyleSheet.create({
  card: { padding: 16, borderRadius: 12, backgroundColor: "#fff", marginBottom: 12, shadowColor: "#000", shadowOpacity: 0.06, shadowOffset: { width: 0, height: 2 }, shadowRadius: 6, elevation: 2 },
  title: { fontSize: 18, fontWeight: "600" },
  badge: { fontWeight: "600" },
});
