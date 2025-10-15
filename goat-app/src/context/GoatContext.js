
import { createContext, useContext, useEffect, useMemo, useState } from "react";
import * as storage from "../lib/storage";
import { nanoid } from "nanoid/non-secure"; // safe for Expo

const GoatContext = createContext();

export function GoatProvider({ children }) {
  const [goats, setGoats] = useState([]);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    (async () => {
      const saved = await storage.loadGoats();
      setGoats(saved);
      setLoaded(true);
    })();
  }, []);

  useEffect(() => { if (loaded) storage.saveGoats(goats); }, [goats, loaded]);

  const addGoat = (g) => {
    const id = g.id?.trim() || nanoid(8);
    const newGoat = { ...g, id, createdAt: Date.now() };
    setGoats((prev) => [newGoat, ...prev]);
    return id;
  };

  const updateGoat = (id, patch) => setGoats((prev) => prev.map((g) => (g.id === id ? { ...g, ...patch } : g)));
  const removeGoat = (id) => setGoats((prev) => prev.filter((g) => g.id !== id));

  const value = useMemo(() => ({ goats, addGoat, updateGoat, removeGoat, loaded }), [goats, loaded]);
  return <GoatContext.Provider value={value}>{children}</GoatContext.Provider>;
}

export const useGoats = () => useContext(GoatContext);
