      const CAMERA_COLD_MESSAGE =
        "Cameras are below the minimum operating temperature. Please wait for the heating system.";
      const DESCRIPTION_OPTIONS = {
        goat: ["meat", "dairy", "cross"],
        lamb: ["lamb", "ewe"],
      };
      const GRADE_OPTIONS = [
        "CAB Prime",
        "Reserve",
        "Prime",
        "Choice",
        "Select",
        "No Roll",
      ];

      // THEME
      function initTheme() {
        return HerdTheme.init("themeBtn");
      }
      function toggleTheme() {
        return HerdTheme.toggle("themeBtn");
      }
      function updateThemeIcon(t) {
        HerdTheme.updateButton("themeBtn", t);
      }
      initTheme();

      // UTILITY
      function esc(s) {
        if (!s) return "";
        const d = document.createElement("div");
        d.textContent = s;
        return d.innerHTML;
      }
      function highlight(text, query) {
        if (!query || !text) return esc(text || "");
        const safe = esc(text),
          lower = safe.toLowerCase(),
          q = query.toLowerCase();
        let result = "",
          last = 0,
          idx;
        while ((idx = lower.indexOf(q, last)) !== -1) {
          result +=
            safe.slice(last, idx) +
            '<span class="match">' +
            safe.slice(idx, idx + q.length) +
            "</span>";
          last = idx + q.length;
        }
        return result + safe.slice(last);
      }
      function fmtDate(d) {
        if (!d) return "—";
        return new Date(d).toLocaleDateString("en-US", {
          month: "short",
          day: "numeric",
          year: "numeric",
        });
      }
      function todayISO() {
        return new Date().toISOString().split("T")[0];
      }
      function wt(v) {
        return v != null ? parseFloat(v).toFixed(1) : "—";
      }
      const GRADE_ORDER = {
        "CAB Prime": 1,
        Reserve: 1,
        Prime: 2,
        Choice: 3,
        Select: 4,
        "No Roll": 5,
      };
      function gradeBadge(g) {
        if (!g) return '<span class="badge badge-ungraded">Ungraded</span>';
        return (
          '<span class="badge ' + gradeClass(g) + '">' + esc(g) + "</span>"
        );
      }
      function gradeClass(g) {
        if (!g) return "badge-ungraded";
        const l = g.toLowerCase().replace(/\s+/g, "");
        if (l === "reserve") return "badge-cabprime";
        if (l === "cabprime") return "badge-cabprime";
        if (l === "prime") return "badge-prime";
        if (l === "choice") return "badge-choice";
        if (l === "select") return "badge-select";
        if (l === "noroll") return "badge-noroll";
        return "badge-ungraded";
      }
      function typeBadge(t) {
        return (
          '<span class="badge type-' +
          t +
          '">' +
          t.charAt(0).toUpperCase() +
          t.slice(1) +
          "</span>"
        );
      }
      function closeModal(id) {
        const el = document.getElementById(id);
        el.classList.remove("open");
        el.classList.remove("stacked");
        if (id === "modalAddProvider") _inlineProviderSelect = null;
      }
      function openModal(id) {
        document.getElementById(id).classList.add("open");
      }
      async function fetchNextGlobalId() {
        try {
          const data = await dbFetch("/animals/next");
          _cachedNextId = data.next_serial_id || data.next_id || data;
          return _cachedNextId;
        } catch (err) {
          _cachedNextId = allAnimals.length
            ? Math.max(...allAnimals.map((a) => a.serial_id)) + 1
            : 1;
          return _cachedNextId;
        }
      }
      function nextGlobalId() {
        return (
          _cachedNextId ||
          (allAnimals.length
            ? Math.max(...allAnimals.map((a) => a.serial_id)) + 1
            : 1)
        );
      }
      function populateProviderSelect(el, selectedId) {
        el.innerHTML =
          '<option value="">None</option><option value="__new__">+ Add new provider...</option>' +
          allProviders
            .map(
              (p) =>
                '<option value="' +
                p.id +
                '"' +
                (p.id == selectedId ? " selected" : "") +
                ">" +
                esc(p.name) +
                "</option>",
            )
            .join("");
        const handler = function () {
          if (this.value === "__new__") {
            this.value = selectedId || "";
            openAddProviderInline(this);
          }
        };
        el.removeEventListener("change", el._provHandler);
        el._provHandler = handler;
        el.addEventListener("change", handler);
      }
      function openAddProviderInline(selectEl) {
        _inlineProviderSelect = selectEl;
        document.getElementById("modalAddProvider").classList.add("stacked");
        openModal("modalAddProvider");
        [
          "provNameInput",
          "provCompInput",
          "provPhoneInput",
          "provEmailInput",
          "provAddrInput",
        ].forEach((id) => {
          document.getElementById(id).value = "";
          document.getElementById(id).classList.remove("error");
        });
        document
          .querySelectorAll("#modalAddProvider .field-error")
          .forEach((e) => e.classList.remove("visible"));
        setTimeout(() => document.getElementById("provNameInput").focus(), 100);
      }

      // AUTH + INIT

      function openLightbox(slot) {
        HerdUI.openLightbox(slot, {
          fallbackName: "debug",
          extension: ".png",
        });
      }
      function closeLightbox(e) {
        HerdUI.closeLightbox(e);
      }
      HerdUI.bindLightboxEscape();

      // ============================================================

      function showToast(type, msg) {
        HerdUI.showToast(type, msg, "toast");
      }

      // KEYBOARD + MODAL OVERLAY CLICKS
      document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") {
          const stacked = document.querySelector(".modal-overlay.open.stacked");
          if (stacked) {
            stacked.classList.remove("open", "stacked");
            _inlineProviderSelect = null;
            return;
          }
          const topModal = document.querySelector(".modal-overlay.open");
          if (topModal) topModal.classList.remove("open");
        }
      });
      document.querySelectorAll(".modal-overlay").forEach((m) => {
        m.addEventListener("click", function (e) {
          if (e.target === this) this.classList.remove("open");
        });
      });

      // ============================================================
