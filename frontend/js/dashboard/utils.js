      const CAMERA_COLD_MESSAGE =
        "Cameras are below the minimum operating temperature. Please wait for the heating system.";
      const DESCRIPTION_OPTIONS = {
        goat: ["meat", "dairy", "cross"],
        lamb: ["lamb", "ewe"],
      };
      const GRADE_OPTIONS = [
        "Reserve / CAB Prime",
        "Prime",
        "CAB Choice",
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
        "Reserve / CAB Prime": 1,
        Prime: 2,
        "CAB Choice": 3,
        Choice: 4,
        Select: 5,
        "No Roll": 6,
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
        if (l === "reserve/cabprime") return "badge-cabprime";
        if (l === "cabprime") return "badge-cabprime";
        if (l === "cabchoice") return "badge-choice";
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
        if (
          id === "modalGradeAnnotation" &&
          typeof closeGradeAnnotationModal === "function" &&
          _gradeAnnotationContext
        ) {
          closeGradeAnnotationModal();
          return;
        }
        const el = document.getElementById(id);
        el.classList.remove("open");
        el.classList.remove("stacked");
        if (id === "modalAddProvider") _inlineProviderSelect = null;
      }
      function openModal(id) {
        document.getElementById(id).classList.add("open");
      }
      async function fetchNextGlobalId() {
        if (
          _pendingNewAnimalSession &&
          _pendingNewAnimalSession.next_serial_id != null
        ) {
          _cachedNextId = _pendingNewAnimalSession.next_serial_id;
          return _cachedNextId;
        }
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
      const ACTIVE_CAPTURE_SESSION_KEY =
        "herdsync_active_new_animal_capture_session";
      function _readActiveCaptureSessionMarker() {
        try {
          const raw = localStorage.getItem(ACTIVE_CAPTURE_SESSION_KEY);
          return raw ? JSON.parse(raw) : null;
        } catch (err) {
          return null;
        }
      }
      function _writeActiveCaptureSessionMarker(session) {
        if (!session || session.status !== "capturing" || !session.id) {
          localStorage.removeItem(ACTIVE_CAPTURE_SESSION_KEY);
          return;
        }
        localStorage.setItem(
          ACTIVE_CAPTURE_SESSION_KEY,
          JSON.stringify({
            id: session.id,
            username: _currentUser?.username || null,
          }),
        );
      }
      function _shouldDiscardFreshLoginCaptureSession(session) {
        if (!session || session.status !== "capturing" || !_currentUser?.username) {
          return false;
        }
        if (session.created_by && session.created_by !== _currentUser.username) {
          return false;
        }
        const marker = _readActiveCaptureSessionMarker();
        return !marker || marker.id !== session.id || marker.username !== _currentUser.username;
      }
      function syncPendingNewAnimalSession(session) {
        _pendingNewAnimalSession = session || null;
        _gradeSessionId = session ? session.id : null;
        _gradeAnalysisKey = session ? session.analysis_key : null;
        if (session && session.next_serial_id != null) {
          _cachedNextId = session.next_serial_id;
          nextGradeId = session.next_serial_id;
        }
        _writeActiveCaptureSessionMarker(session);
        return _pendingNewAnimalSession;
      }
      function hasPendingNewAnimalSession() {
        return (
          !!_pendingNewAnimalSession &&
          ["capturing", "review_pending", "finalizing"].includes(
            _pendingNewAnimalSession.status,
          )
        );
      }
      function pendingNewAnimalGateMessage() {
        return (
          "Finish or discard the pending new-animal grade before creating another animal."
        );
      }
      async function loadPendingNewAnimalSession() {
        try {
          const session = await dbFetch("/grading/sessions/pending");
          if (_shouldDiscardFreshLoginCaptureSession(session)) {
            await discardNewAnimalSession(session.id);
            return null;
          }
          return syncPendingNewAnimalSession(session);
        } catch (err) {
          if (!String(err.message || "").startsWith("404")) {
            throw err;
          }
          syncPendingNewAnimalSession(null);
          return null;
        }
      }
      async function createOrResumeNewAnimalSession(body) {
        const resp = await HerdAuth.fetch("/db/grading/sessions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (!resp.ok) {
          const err = await resp.json().catch(() => ({}));
          throw new Error(err.detail || err.message || "Could not start grading session");
        }
        const session = await resp.json();
        return syncPendingNewAnimalSession(session);
      }
      async function updateNewAnimalSession(sessionId, body) {
        const resp = await HerdAuth.fetch("/db/grading/sessions/" + sessionId, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (!resp.ok) {
          const err = await resp.json().catch(() => ({}));
          throw new Error(err.detail || err.message || "Could not update grading session");
        }
        const session = await resp.json();
        return syncPendingNewAnimalSession(session);
      }
      async function discardNewAnimalSession(sessionId) {
        if (!sessionId) return null;
        const resp = await HerdAuth.fetch(
          "/db/grading/sessions/" + sessionId + "/discard",
          { method: "POST" },
        );
        if (!resp.ok) {
          const err = await resp.json().catch(() => ({}));
          throw new Error(err.detail || err.message || "Could not discard grading session");
        }
        const session = await resp.json();
        syncPendingNewAnimalSession(null);
        return session;
      }
      async function finalizeNewAnimalSession(sessionId, body) {
        const resp = await HerdAuth.fetch(
          "/db/grading/sessions/" + sessionId + "/finalize",
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body || {}),
          },
        );
        if (!resp.ok) {
          const err = await resp.json().catch(() => ({}));
          throw new Error(err.detail || err.message || "Could not save new animal grade");
        }
        syncPendingNewAnimalSession(null);
        return resp.json();
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
          if (topModal) {
            if (
              topModal.id === "modalGradeAnnotation" &&
              typeof closeGradeAnnotationModal === "function" &&
              _gradeAnnotationContext
            ) {
              closeGradeAnnotationModal();
            } else {
              topModal.classList.remove("open");
            }
          }
        }
      });
      document.querySelectorAll(".modal-overlay").forEach((m) => {
        m.addEventListener("click", function (e) {
          if (e.target === this) {
            if (
              this.id === "modalGradeAnnotation" &&
              typeof closeGradeAnnotationModal === "function" &&
              _gradeAnnotationContext
            ) {
              closeGradeAnnotationModal();
            } else {
              this.classList.remove("open");
            }
          }
        });
      });

      // ============================================================
