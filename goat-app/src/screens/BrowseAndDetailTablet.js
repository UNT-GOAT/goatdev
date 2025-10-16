
import { useMemo, useState } from "react";
import { View, TextInput, FlatList, StyleSheet, Text } from "react-native";
import { useGoats } from "../context/GoatContext";
import GoatCard from "../components/GoatCard";
import GoatDetail from "./GoatDetailInline";

export default function BrowseAndDetailTablet() {
  const { goats } = useGoats();
  const [q, setQ] = useState("");
  const [selectedId, setSelectedId] = useState(null);

  const data = useMemo(() => {
    const term = q.trim().toLowerCase();
    if (!term) return goats;
    return goats.filter((g) =>
      [g.id, g.grade, g.notes]?.some((v) => String(v || "").toLowerCase().includes(term)) ||
      String(g.weight).includes(term) || String(g.height).includes(term) || String(g.length).includes(term)
    );
  }, [q, goats]);

  return (
    <View style={styles.split}>
      <View style={styles.left}>
        <TextInput style={styles.search} placeholder="Search…" value={q} onChangeText={setQ} />
        <FlatList
          data={data}
          keyExtractor={(item) => String(item.id)}
          renderItem={({ item }) => <GoatCard goat={item} onPress={() => setSelectedId(item.id)} />}
          ListEmptyComponent={<Text style={{ textAlign: "center", marginTop: 40 }}>No goats yet.</Text>}
        />
      </View>
      <View style={styles.right}>
        {selectedId ? <GoatDetail id={selectedId} /> : <Text style={{ padding: 16 }}>Select a goat to view details</Text>}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  split: { flex: 1, flexDirection: "row" },
  left: { width: 380, borderRightWidth: 1, borderRightColor: "#eee", padding: 16 },
  right: { flex: 1, padding: 16 },
  search: { borderWidth: 1, borderColor: "#ddd", borderRadius: 10, padding: 12, backgroundColor: "#fff", marginBottom: 12 },
});
