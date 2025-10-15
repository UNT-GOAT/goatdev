
import { useState } from "react";
import { View, Text, TextInput, Button, Alert, StyleSheet, ScrollView } from "react-native";
import { useGoats } from "../context/GoatContext";
import { isPositiveNumber } from "../utils/validators";

export default function EnterGoatScreen() {
  const { addGoat } = useGoats();
  const [form, setForm] = useState({ id: "", weight: "", height: "", length: "", grade: "", notes: "" });
  const onChange = (k, v) => setForm((f) => ({ ...f, [k]: v }));

  const submit = () => {
    if (!isPositiveNumber(form.weight) || !isPositiveNumber(form.height) || !isPositiveNumber(form.length)) {
      Alert.alert("Invalid input", "Weight, height, and length must be positive numbers.");
      return;
    }
    const payload = { id: form.id.trim() || undefined, weight: Number(form.weight), height: Number(form.height), length: Number(form.length), grade: form.grade.trim() || undefined, notes: form.notes.trim() || undefined };
    const newId = addGoat(payload);
    Alert.alert("Saved", `Goat #${newId} saved.`);
    setForm({ id: "", weight: "", height: "", length: "", grade: "", notes: "" });
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.header}>Enter Goat</Text>
      <LabeledInput label="Goat ID (optional)" value={form.id} onChangeText={(t) => onChange("id", t)} />
      <LabeledInput label="Weight (kg)" keyboardType="numeric" value={form.weight} onChangeText={(t) => onChange("weight", t)} />
      <LabeledInput label="Height (cm)" keyboardType="numeric" value={form.height} onChangeText={(t) => onChange("height", t)} />
      <LabeledInput label="Length (cm)" keyboardType="numeric" value={form.length} onChangeText={(t) => onChange("length", t)} />
      <LabeledInput label="Grade (optional)" value={form.grade} onChangeText={(t) => onChange("grade", t)} placeholder="e.g., A/B/C" />
      <Text style={styles.label}>Notes (optional)</Text>
      <TextInput style={[styles.input, styles.notes]} multiline value={form.notes} onChangeText={(t) => onChange("notes", t)} />
      <Button title="Save Goat" onPress={submit} />
    </ScrollView>
  );
}
function LabeledInput({ label, ...props }) {
  return (
    <View style={{ marginBottom: 14 }}>
      <Text style={styles.label}>{label}</Text>
      <TextInput style={styles.input} {...props} />
    </View>
  );
}
const styles = StyleSheet.create({
  container: { padding: 20 },
  header: { fontSize: 26, fontWeight: "700", marginBottom: 16 },
  label: { fontWeight: "600", marginBottom: 6 },
  input: { borderWidth: 1, borderColor: "#ddd", borderRadius: 12, padding: 12, backgroundColor: "#fff" },
  notes: { height: 120, textAlignVertical: "top" },
});
