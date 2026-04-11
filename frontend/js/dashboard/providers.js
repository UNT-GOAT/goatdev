      function filterProviders() {
        const q = (document.getElementById("provSearch")?.value || "")
          .trim()
          .toLowerCase();
        document
          .getElementById("provClear")
          .classList.toggle("visible", q.length > 0);
        let results = allProviders.filter((p) => {
          if (provFilter === "active" && p.status !== "active") return false;
          if (!q) return true;
          return [p.name, p.company, p.phone, p.email, p.address]
            .join(" ")
            .toLowerCase()
            .includes(q);
        });
        document.getElementById("provCount").textContent =
          results.length + " of " + allProviders.length;
        renderProviderGrid(results, q);
      }
      function clearProvSearch() {
        document.getElementById("provSearch").value = "";
        document.getElementById("provClear").classList.remove("visible");
        filterProviders();
        document.getElementById("provSearch").focus();
      }
      function setProvFilter(chip, v) {
        chip
          .closest(".filter-row")
          .querySelectorAll(".filter-chip")
          .forEach((c) => c.classList.remove("active"));
        chip.classList.add("active");
        provFilter = v;
        filterProviders();
      }
      function renderProviderGrid(providers, q) {
        const grid = document.getElementById("provGrid");
        if (!providers.length) {
          grid.innerHTML = '<div class="no-results">No providers match</div>';
          return;
        }
        grid.innerHTML = providers
          .map((p) => {
            const cnt = allAnimals.filter((a) => a.prov_id === p.id).length;
            const headBadge =
              cnt > 0
                ? '<span class="badge badge-prime" style="font-size:10px">' +
                  cnt +
                  " head</span>"
                : '<span style="font-size:10px;color:var(--text-muted);background:var(--bg-raised);padding:3px 8px;border-radius:20px">0 head</span>';
            const details = [
              p.company ? highlight(p.company, q) : null,
              p.phone ? highlight(p.phone, q) : null,
              p.email
                ? '<span style="color:var(--text-faint)">' +
                  highlight(p.email, q) +
                  "</span>"
                : null,
            ]
              .filter(Boolean)
              .join(" · ");
            return (
              '<div class="row-item" style="padding:12px 16px;cursor:pointer;transition:background .12s" onclick="openEditProvider(' +
              p.id +
              ')" onmouseenter="this.style.background=\'var(--accent-dim)\'" onmouseleave="this.style.background=\'transparent\'"><div class="flex justify-between items-center" style="margin-bottom:3px"><div style="font-size:14px;font-weight:600;color:var(--text-primary)">' +
              highlight(p.name, q) +
              '</div><div class="flex items-center gap-8"><span style="font-size:9px;color:var(--text-faint);text-transform:uppercase;letter-spacing:1px">' +
              (p.status || "active") +
              "</span>" +
              headBadge +
              "</div></div>" +
              (details
                ? '<div style="font-size:11px;color:var(--text-muted)">' +
                  details +
                  "</div>"
                : "") +
              (p.address
                ? '<div style="font-size:10px;color:var(--text-faint);margin-top:2px">' +
                  highlight(p.address, q) +
                  "</div>"
                : "") +
              "</div>"
            );
          })
          .join("");
      }
      function openAddProvider() {
        _inlineProviderSelect = null;
        document.getElementById("modalAddProvider").classList.remove("stacked");
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
      async function saveProvider() {
        let ok = true;
        const name = document.getElementById("provNameInput").value.trim(),
          phone = document.getElementById("provPhoneInput").value.trim(),
          email = document.getElementById("provEmailInput").value.trim();
        if (!name) {
          document.getElementById("provNameInput").classList.add("error");
          document.getElementById("provNameErr").classList.add("visible");
          ok = false;
        } else {
          document.getElementById("provNameInput").classList.remove("error");
          document.getElementById("provNameErr").classList.remove("visible");
        }
        if (!phone) {
          document.getElementById("provPhoneInput").classList.add("error");
          document.getElementById("provPhoneErr").classList.add("visible");
          ok = false;
        } else {
          document.getElementById("provPhoneInput").classList.remove("error");
          document.getElementById("provPhoneErr").classList.remove("visible");
        }
        if (email && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
          document.getElementById("provEmailInput").classList.add("error");
          document.getElementById("provEmailErr").classList.add("visible");
          ok = false;
        } else {
          document.getElementById("provEmailInput").classList.remove("error");
          document.getElementById("provEmailErr").classList.remove("visible");
        }
        if (!ok) return;
        try {
          const r = await HerdAuth.fetch("/db/providers", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              name,
              company:
                document.getElementById("provCompInput").value.trim() || null,
              phone,
              email: email || null,
              address:
                document.getElementById("provAddrInput").value.trim() || null,
              status: "active",
            }),
          });
          if (!r.ok) throw new Error(r.status);
          const newProv = await r.json();
          const wasInline = !!_inlineProviderSelect;
          closeModal("modalAddProvider");
          await loadProviders();
          filterProviders();
          updateDashboard();
          if (wasInline) {
            const newId =
              newProv.id || allProviders[allProviders.length - 1]?.id;
            document
              .querySelectorAll(
                "#addAnimalProv,#batchAnimalProv,#editAnimalProv,#gradeProv",
              )
              .forEach((sel) => {
                populateProviderSelect(sel, newId);
                sel.value = String(newId);
              });
          }
          showToast("success", "Provider added — " + name);
        } catch (err) {
          showToast("error", "Save failed: " + err.message);
        }
      }
      function openEditProvider(id) {
        const p = allProviders.find((x) => x.id === id);
        if (!p) return;
        openModal("modalEditProvider");
        document.getElementById("editProvId").value = p.id;
        document.getElementById("editProvName").value = p.name || "";
        document.getElementById("editProvComp").value = p.company || "";
        document.getElementById("editProvPhone").value = p.phone || "";
        document.getElementById("editProvEmail").value = p.email || "";
        document.getElementById("editProvAddr").value = p.address || "";
        document.getElementById("editProvStatus").value = p.status || "active";
        const cnt = allAnimals.filter((a) => a.prov_id === p.id).length;
        document.getElementById("editProvHeadCount").textContent =
          cnt + " head";
        document.getElementById("editProvSince").textContent = fmtDate(
          p.active_since || p.created_at,
        );
      }
      async function saveEditProvider() {
        const id = parseInt(document.getElementById("editProvId").value);
        const name = document.getElementById("editProvName").value.trim();
        const phone = document.getElementById("editProvPhone").value.trim();
        if (!name) {
          showToast("error", "Name is required");
          return;
        }
        if (!phone) {
          showToast("error", "Phone is required");
          return;
        }
        const email = document.getElementById("editProvEmail").value.trim();
        try {
          const r = await HerdAuth.fetch("/db/providers/" + id, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              name,
              phone,
              email: email || null,
              company:
                document.getElementById("editProvComp").value.trim() || null,
              address:
                document.getElementById("editProvAddr").value.trim() || null,
              status: document.getElementById("editProvStatus").value,
            }),
          });
          if (!r.ok) {
            const err = await r.json().catch(() => ({}));
            throw new Error(err.detail || "Failed: " + r.status);
          }
          closeModal("modalEditProvider");
          await loadProviders();
          filterProviders();
          updateDashboard();
          showToast("success", "Provider updated — " + name);
        } catch (err) {
          showToast("error", err.message);
        }
      }
      function deleteProvider() {
        const id = parseInt(document.getElementById("editProvId").value);
        const p = allProviders.find((x) => x.id === id);
        if (!p) return;
        const cnt = allAnimals.filter((a) => a.prov_id === p.id).length;
        document.getElementById("deleteProvMsg").textContent =
          "Permanently delete " +
          p.name +
          "?" +
          (cnt > 0
            ? " " + cnt + " animals will be unlinked from this provider."
            : "");
        openModal("modalDeleteProvider");
      }
      async function confirmDeleteProvider() {
        const id = parseInt(document.getElementById("editProvId").value);
        closeModal("modalDeleteProvider");
        try {
          const r = await HerdAuth.fetch("/db/providers/" + id, {
            method: "DELETE",
          });
          if (!r.ok) {
            const err = await r.json().catch(() => ({}));
            throw new Error(err.detail || "Delete failed: " + r.status);
          }
          closeModal("modalEditProvider");
          showToast("success", "Provider deleted");
          await Promise.all([
            loadProviders().catch(() => {}),
            loadGoats().catch(() => {}),
            loadChickens().catch(() => {}),
            loadLambs().catch(() => {}),
          ]);
          filterProviders();
          filterAnimals();
          updateDashboard();
        } catch (err) {
          showToast("error", err.message);
        }
      }

      // GRADING
