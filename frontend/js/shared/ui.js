(function (global) {
  let toastTimer = null;

  global.HerdUI = {
    showToast(type, msg, toastId) {
      const toast = document.getElementById(toastId || "toast");
      if (!toast) return;
      toast.className = "toast " + type;
      toast.querySelector(".toast-icon").textContent =
        type === "success" ? "✓" : "✕";
      toast.querySelector(".toast-msg").textContent = msg;
      clearTimeout(toastTimer);
      requestAnimationFrame(() => toast.classList.add("show"));
      toastTimer = setTimeout(() => toast.classList.remove("show"), 3500);
    },

    openLightbox(slot, options) {
      const img = slot.querySelector("img");
      if (!img || !img.src) return;
      const config = options || {};
      const lightboxId = config.lightboxId || "lightbox";
      const imageId = config.imageId || "lightboxImg";
      const downloadId = config.downloadId || "lightboxDl";
      const fallbackName = config.fallbackName || "image";
      const extension = config.extension || ".png";

      document.getElementById(imageId).src = img.src;
      document.getElementById(downloadId).href = img.src;
      document.getElementById(downloadId).download =
        (img.alt || fallbackName) + extension;
      document.getElementById(lightboxId).classList.add("open");
    },

    closeLightbox(event, options) {
      if (event.target.tagName === "A") return;
      const config = options || {};
      document
        .getElementById(config.lightboxId || "lightbox")
        .classList.remove("open");
    },

    bindLightboxEscape(options) {
      const config = options || {};
      const lightboxId = config.lightboxId || "lightbox";
      document.addEventListener(
        "keydown",
        function (event) {
          const lightbox = document.getElementById(lightboxId);
          if (event.key === "Escape" && lightbox.classList.contains("open")) {
            lightbox.classList.remove("open");
            event.stopImmediatePropagation();
          }
        },
        true,
      );
    },
  };
})(window);
