
import { useMemo, useState } from "react";
import { View, TextInput, FlatList, StyleSheet, Text } from "react-native";
import { useGoats } from "../context/GoatContext";
import GoatCard from "../components/GoatCard";

export default function BrowseGoatsScreen({ navigation }) {
  const { goats } = useGoats();
  const [q, setQ] = useState("");

  const data = useMemo(() => {
    const term = q.trim().toLowerCase();
    if (!term) return goats;
    return goats.filter((g) =>
      [g.id, g.grade, g.notes]?.some((v) => String(v || "").toLowerCase().includes(term)) ||
      String(g.weight).includes(term) || String(g.height).includes(term) || String(g.length).includes(term)
    );
  }, [q, goats]);

  return (
    <View style={styles.container}>
      <TextInput style={styles.search} placeholder="Search by ID, grade, note, or number…" value={q} onChangeText={setQ} />
      <FlatList
        data={data}
        keyExtractor={(item) => String(item.id)}
        renderItem={({ item }) => <GoatCard goat={item} onPress={() => navigation.navigate("GoatDetail", { id: item.id })} />}
        ListEmptyComponent={<Text style={{ textAlign: "center", marginTop: 40 }}>No goats yet.</Text>}
      />
    </View>
  );
}
const styles = StyleSheet.create({
  container: { flex: 1, padding: 16 },
  search: { borderWidth: 1, borderColor: "#ddd", borderRadius: 12, padding: 12, backgroundColor: "#fff", marginBottom: 12 },
});
