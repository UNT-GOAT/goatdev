(function (global) {
  const STORAGE_KEY = "herdsync-theme";

  global.HerdTheme = {
    init(buttonId) {
      const theme = localStorage.getItem(STORAGE_KEY) || "dark";
      document.documentElement.dataset.theme = theme;
      this.updateButton(buttonId, theme);
      return theme;
    },

    toggle(buttonId) {
      const next =
        document.documentElement.dataset.theme === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      localStorage.setItem(STORAGE_KEY, next);
      this.updateButton(buttonId, next);
      return next;
    },

    updateButton(buttonId, theme) {
      const button = document.getElementById(buttonId);
      if (!button) return;
      button.textContent = theme === "dark" ? "☀" : "☾";
    },
  };
})(window);
