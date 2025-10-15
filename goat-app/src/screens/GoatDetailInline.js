
import { useMemo, useState } from "react";
import { View, Text, TextInput, Button, Alert, StyleSheet, ScrollView } from "react-native";
import { useGoats } from "../context/GoatContext";
import { isPositiveNumber } from "../utils/validators";

export default function GoatDetail({ id }) {
  const { goats, updateGoat, removeGoat } = useGoats();
  const goat = useMemo(() => goats.find((g) => g.id === id), [goats, id]);
  const [edit, setEdit] = useState(goat || {});

  if (!goat) return <View style={{ padding: 16 }}><Text>Goat not found.</Text></View>;

  const save = () => {
    if (!isPositiveNumber(edit.weight) || !isPositiveNumber(edit.height) || !isPositiveNumber(edit.length)) {
      Alert.alert("Invalid input", "Weight, height, and length must be positive numbers.");
      return;
    }
    updateGoat(goat.id, {
      weight: Number(edit.weight),
      height: Number(edit.height),
      length: Number(edit.length),
      grade: edit.grade?.trim() || undefined,
      notes: edit.notes?.trim() || undefined,
    });
    Alert.alert("Updated", `Goat #${goat.id} updated.`);
  };

  const del = () => {
    Alert.alert("Delete", `Delete goat #${goat.id}?`, [
      { text: "Cancel", style: "cancel" },
      { text: "Delete", style: "destructive", onPress: () => removeGoat(goat.id) },
    ]);
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.header}>Goat #{goat.id}</Text>
      <LabeledInput label="Weight (kg)" keyboardType="numeric" value={String(edit.weight)} onChangeText={(t) => setEdit((e) => ({ ...e, weight: t }))} />
      <LabeledInput label="Height (cm)" keyboardType="numeric" value={String(edit.height)} onChangeText={(t) => setEdit((e) => ({ ...e, height: t }))} />
      <LabeledInput label="Length (cm)" keyboardType="numeric" value={String(edit.length)} onChangeText={(t) => setEdit((e) => ({ ...e, length: t }))} />
      <LabeledInput label="Grade" value={edit.grade || ""} onChangeText={(t) => setEdit((e) => ({ ...e, grade: t }))} />
      <Text style={styles.label}>Notes</Text>
      <TextInput style={[styles.input, styles.notes]} multiline value={edit.notes || ""} onChangeText={(t) => setEdit((e) => ({ ...e, notes: t }))} />
      <View style={{ gap: 10 }}>
        <Button title="Save Changes" onPress={save} />
        <Button title="Delete Goat" color="#b00020" onPress={del} />
      </View>
    </ScrollView>
  );
}

function LabeledInput({ label, ...props }) {
  return (
    <View style={{ marginBottom: 12 }}>
      <Text style={styles.label}>{label}</Text>
      <TextInput style={styles.input} {...props} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { padding: 8 },
  header: { fontSize: 22, fontWeight: "700", marginBottom: 16 },
  label: { fontWeight: "600", marginBottom: 6 },
  input: { borderWidth: 1, borderColor: "#ddd", borderRadius: 10, padding: 10, backgroundColor: "#fff" },
  notes: { height: 120, textAlignVertical: "top" },
});
