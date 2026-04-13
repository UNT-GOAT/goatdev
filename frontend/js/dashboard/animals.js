      function filterAnimals() {
        const q = (document.getElementById("animalSearch")?.value || "")
          .trim()
          .toLowerCase();
        document
          .getElementById("animalClear")
          .classList.toggle("visible", q.length > 0);
        let results = allAnimals.filter((a) => {
          if (animalFilter === "goats" && a.type !== "goat") return false;
          if (animalFilter === "chickens" && a.type !== "chicken") return false;
          if (animalFilter === "lambs" && a.type !== "lamb") return false;
          if (animalFilter === "graded" && !a.grade) return false;
          if (!q) return true;
          return [
            String(a.serial_id),
            a.type,
            a.description,
            a.grade || "ungraded",
            a.providerName,
            String(a.hang_weight || ""),
          ]
            .join(" ")
            .toLowerCase()
            .includes(q);
        });
        results.sort((a, b) => {
          if (animalSort === "id-desc") return b.serial_id - a.serial_id;
          if (animalSort === "id-asc") return a.serial_id - b.serial_id;
          if (animalSort === "weight-desc")
            return (b.hang_weight || 0) - (a.hang_weight || 0);
          if (animalSort === "weight-asc")
            return (a.hang_weight || 0) - (b.hang_weight || 0);
          if (animalSort === "grade")
            return (GRADE_ORDER[a.grade] || 99) - (GRADE_ORDER[b.grade] || 99);
          if (animalSort === "created-desc")
            return new Date(b.created_at || 0) - new Date(a.created_at || 0);
          if (animalSort === "created-asc")
            return new Date(a.created_at || 0) - new Date(b.created_at || 0);
          return 0;
        });
        document.getElementById("animalCount").textContent =
          results.length + " of " + allAnimals.length;
        renderAnimalTable(results, q);
      }
      function clearAnimalSearch() {
        document.getElementById("animalSearch").value = "";
        document.getElementById("animalClear").classList.remove("visible");
        filterAnimals();
        document.getElementById("animalSearch").focus();
      }
      function setAnimalFilter(chip, v) {
        chip
          .closest(".filter-row")
          .querySelectorAll(".filter-chip")
          .forEach((c) => c.classList.remove("active"));
        chip.classList.add("active");
        animalFilter = v;
        filterAnimals();
      }
      function setAnimalSort(chip, v) {
        chip
          .closest(".filter-row")
          .querySelectorAll(".filter-chip")
          .forEach((c) => c.classList.remove("active"));
        chip.classList.add("active");
        animalSort = v;
        filterAnimals();
      }
      function renderAnimalTable(animals, q) {
        const body = document.getElementById("animalTableBody");
        if (!animals.length) {
          body.innerHTML =
            '<div class="no-results">No animals match<div class="hint">Try a different search or filter</div></div>';
          document.getElementById("animalDetail").style.display = "none";
          return;
        }
        body.innerHTML = animals
          .map((a) => {
            const sel =
              a.serial_id === selectedAnimalId && a.type === selectedAnimalType
                ? "background:var(--accent-dim);"
                : "";
            return (
              '<div class="tbl-row" style="grid-template-columns:80px 76px 1fr 72px 94px 1fr 90px;' +
              sel +
              '" onclick="selectAnimal(' +
              a.serial_id +
              ",'" +
              a.type +
              "')\">" +
              "<span style=\"color:var(--text-primary);font-weight:600;font-family:'JetBrains Mono',monospace;font-size:12px\">" +
              highlight(String(a.serial_id), q) +
              "</span>" +
              typeBadge(a.type) +
              '<span style="color:var(--text-secondary);font-size:12px">' +
              highlight(a.description || "—", q) +
              "</span>" +
              "<span style=\"color:var(--text-secondary);font-family:'JetBrains Mono',monospace;font-size:12px\">" +
              wt(a.hang_weight) +
              "</span>" +
              gradeBadge(a.grade) +
              '<span style="color:var(--text-muted);font-size:12px">' +
              highlight(a.providerName, q) +
              "</span>" +
              '<span style="color:var(--text-faint);font-size:10px">' +
              fmtDate(a.created_at) +
              "</span></div>"
            );
          })
          .join("");
      }
      function selectAnimal(sid, type) {
        const a = allAnimals.find(
          (x) => x.serial_id === sid && x.type === type,
        );
        if (!a) return;
        selectedAnimalId = sid;
        selectedAnimalType = type;
        filterAnimals();
        document.getElementById("animalDetail").style.display = "";
        if (window.innerWidth <= 900) document.getElementById("animalDetail").scrollIntoView({ behavior: 'smooth' });
        document.getElementById("detailSerial").textContent = a.serial_id;
        document.getElementById("detailType").innerHTML = typeBadge(a.type);
        document.getElementById("detailDesc").textContent =
          a.description || "—";
        document.getElementById("detailHW").textContent = wt(a.hang_weight);
        document.getElementById("detailLW").textContent = wt(a.live_weight);
        document.getElementById("detailGrade").innerHTML = gradeBadge(a.grade);
        document.getElementById("detailProvider").textContent = a.providerName;
        document.getElementById("detailKill").textContent = fmtDate(
          a.kill_date,
        );
        document.getElementById("detailProcess").textContent = fmtDate(
          a.process_date,
        );
        document.getElementById("detailCreated").textContent = fmtDate(
          a.created_at,
        );
        const acts = document.getElementById("detailActions");
        const animalGrades = allGrades.filter(
          (g) => g.serial_id == a.serial_id,
        );
        const hasGrade = animalGrades.length > 0;
        const gradeLabel = hasGrade ? "Regrade" : "Grade";
        acts.innerHTML =
          (a.type === "goat" || a.type === "lamb"
            ? '<button class="btn btn-accent flex-1 btn-sm" onclick="openGradeForAnimal()">' +
              gradeLabel +
              "</button>"
            : "") +
          (hasGrade
            ? '<button class="btn flex-1 btn-sm" onclick="openAnimalGrades()">Edit Grade</button>'
            : "") +
          '<button class="btn flex-1 btn-sm" onclick="openEditAnimal()">Edit Animal</button><button class="btn btn-danger flex-1 btn-sm" onclick="deleteAnimal()">Delete</button>';
      }
      function openGradeForAnimal() {
        const a = allAnimals.find(
          (x) =>
            x.serial_id === selectedAnimalId && x.type === selectedAnimalType,
        );
        if (!a) return;
        document
          .querySelectorAll(".nav-btn")
          .forEach((b) => b.classList.remove("active"));
        document.querySelector('[data-page="grading"]').classList.add("active");
        document
          .querySelectorAll(".page")
          .forEach((p) => p.classList.remove("active"));
        document.getElementById("page-grading").classList.add("active");
        document.getElementById("pageTitle").textContent = "Grading";
        currentPage = "grading";
        setCamerasForPage("grading");
        _gradeExistingId = a.serial_id;
        document.getElementById("gradeSpecies").value = a.type;
        onGradeSpeciesChange();
        if (a.description)
          document.getElementById("gradeDesc").value = a.description;
        document.getElementById("gradeSerial").value = a.serial_id;
        document.getElementById("gradeSerialHint").textContent =
          "Grading existing " + a.type + " #" + a.serial_id;
        document.getElementById("gradeSerialHint").className =
          "field-hint valid";
        if (a.live_weight)
          document.getElementById("gradeWeight").value = a.live_weight;
        populateProviderSelect(document.getElementById("gradeProv"), a.prov_id);
        document.getElementById("gradeKillDate").value = a.kill_date
          ? a.kill_date.split("T")[0]
          : todayISO();
        if (piConnected) document.getElementById("gradeBtn").disabled = false;
        document.getElementById("gradeWeight").focus();
      }
      function openAnimalGrades() {
        const a = allAnimals.find(
          (x) =>
            x.serial_id === selectedAnimalId && x.type === selectedAnimalType,
        );
        if (!a) return;
        const animalGrades = allGrades.filter(
          (g) => g.serial_id == a.serial_id,
        );
        if (!animalGrades.length) {
          showToast("error", "No grades for this animal");
          return;
        }
        viewGradeDetail(animalGrades[0].id);
      }

      // ADD ANIMAL
      function openAddAnimal() {
        openModal("modalAddAnimal");
        switchAddAnimalTab(
          "single",
          document.querySelector("#modalAddAnimal .tab-switch-btn"),
        );
        populateProviderSelect(document.getElementById("addAnimalProv"));
        populateProviderSelect(document.getElementById("batchAnimalProv"));
        onAddAnimalTypeChange();
        document.getElementById("batchAnimalCount").value = "";
        document.getElementById("batchStatus").textContent = "";
        [
          "addAnimalHW",
          "addAnimalLW",
          "addAnimalKill",
          "addAnimalProc",
          "batchAnimalKill",
          "batchAnimalProc",
        ].forEach((id) => {
          document.getElementById(id).value = "";
        });
      }
      function switchAddAnimalTab(tab, btn) {
        document
          .querySelectorAll("#modalAddAnimal .tab-switch-btn")
          .forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        document
          .querySelectorAll("#modalAddAnimal .tab-pane")
          .forEach((p) => p.classList.remove("active"));
        document
          .getElementById(
            tab === "single" ? "addAnimalSingle" : "addAnimalMultiple",
          )
          .classList.add("active");
      }
      async function onAddAnimalTypeChange() {
        const type = document.getElementById("addAnimalType").value;
        const hint = document.getElementById("addAnimalSerialHint");
        hint.textContent = "Fetching next ID...";
        const nextId = await fetchNextGlobalId();
        document.getElementById("addAnimalSerial").value = nextId;
        hint.textContent = "Auto-assigned";
        const wrap = document.getElementById("addAnimalDescWrap");
        const descSel = document.getElementById("addAnimalDesc");
        if (DESCRIPTION_OPTIONS[type]) {
          wrap.style.display = "";
          descSel.innerHTML = DESCRIPTION_OPTIONS[type]
            .map(
              (o) =>
                '<option value="' +
                o +
                '">' +
                o.charAt(0).toUpperCase() +
                o.slice(1) +
                "</option>",
            )
            .join("");
        } else {
          wrap.style.display = "none";
        }
        const gradeWrap = document.getElementById("addAnimalGradeWrap");
        gradeWrap.style.display =
          type === "goat" || type === "lamb" ? "" : "none";
        document.getElementById("addAnimalGrade").value = "";
      }
      async function saveAnimal() {
        const isMultiple = document
          .getElementById("addAnimalMultiple")
          .classList.contains("active");
        const btn = document.getElementById("addAnimalSaveBtn");
        btn.disabled = true;
        btn.textContent = "Saving...";
        try {
          if (isMultiple) await saveBatchAnimals();
          else await saveSingleAnimal();
          closeModal("modalAddAnimal");
          await Promise.all([
            loadGoats().catch(() => {}),
            loadChickens().catch(() => {}),
            loadLambs().catch(() => {}),
          ]);
          updateDashboard();
          filterAnimals();
          computeNextGradeId();
        } catch (err) {
          showToast("error", err.message);
        } finally {
          btn.disabled = false;
          btn.textContent = "Save";
        }
      }
      async function saveSingleAnimal() {
        const type = document.getElementById("addAnimalType").value;
        const body = {};
        if (DESCRIPTION_OPTIONS[type])
          body.description = document.getElementById("addAnimalDesc").value;
        const grade = document.getElementById("addAnimalGrade").value;
        if (grade) body.grade = grade;
        const hw = parseFloat(document.getElementById("addAnimalHW").value);
        if (hw > 0) body.hang_weight = hw;
        const lw = parseFloat(document.getElementById("addAnimalLW").value);
        if (lw > 0) body.live_weight = lw;
        const prov = document.getElementById("addAnimalProv").value;
        if (prov) body.prov_id = parseInt(prov);
        const kd = document.getElementById("addAnimalKill").value;
        if (kd) body.kill_date = kd;
        const pd = document.getElementById("addAnimalProc").value;
        if (pd) body.process_date = pd;
        const resp = await HerdAuth.fetch("/db/" + type + "s", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (!resp.ok) {
          const err = await resp.json().catch(() => ({}));
          throw new Error(err.detail || "Failed: " + resp.status);
        }
        const created = await resp.json();
        showToast(
          "success",
          type.charAt(0).toUpperCase() +
            type.slice(1) +
            " #" +
            created.serial_id +
            " added",
        );
      }
      async function saveBatchAnimals() {
        const type = document.getElementById("batchAnimalType").value,
          count = parseInt(document.getElementById("batchAnimalCount").value);
        if (!count || count < 1 || count > 500)
          throw new Error("Enter count 1-500");
        const prov = document.getElementById("batchAnimalProv").value,
          kd = document.getElementById("batchAnimalKill").value,
          pd = document.getElementById("batchAnimalProc").value;
        const status = document.getElementById("batchStatus");
        let added = 0,
          failed = 0;
        for (let i = 0; i < count; i++) {
          status.textContent = "Adding " + (i + 1) + "/" + count + "...";
          status.className = "field-hint";
          try {
            const body = {};
            if (prov) body.prov_id = parseInt(prov);
            if (kd) body.kill_date = kd;
            if (pd) body.process_date = pd;
            const r = await HerdAuth.fetch("/db/" + type + "s", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(body),
            });
            if (r.ok) added++;
            else failed++;
          } catch (e) {
            failed++;
          }
        }
        status.textContent =
          "Done: " +
          added +
          " added" +
          (failed ? " (" + failed + " failed)" : "");
        status.className = "field-hint " + (failed ? "invalid" : "valid");
        showToast("success", added + " " + type + "s added");
      }

      // EDIT ANIMAL
      function openEditAnimal() {
        const a = allAnimals.find(
          (x) =>
            x.serial_id === selectedAnimalId && x.type === selectedAnimalType,
        );
        if (!a) return;
        openModal("modalEditAnimal");
        document.getElementById("editAnimalSerial").value = a.serial_id;
        document.getElementById("editAnimalType").value = a.type;
        document.getElementById("editAnimalHW").value = a.hang_weight || "";
        document.getElementById("editAnimalLW").value = a.live_weight || "";
        const descWrap = document.getElementById("editAnimalDescWrap"),
          descSel = document.getElementById("editAnimalDesc");
        if (a.type === "goat") {
          descWrap.style.display = "";
          descSel.innerHTML = DESCRIPTION_OPTIONS.goat
            .map(
              (o) =>
                '<option value="' +
                o +
                '">' +
                o.charAt(0).toUpperCase() +
                o.slice(1) +
                "</option>",
            )
            .join("");
          descSel.value = a.description || "";
        } else if (a.type === "lamb") {
          descWrap.style.display = "";
          descSel.innerHTML = DESCRIPTION_OPTIONS.lamb
            .map(
              (o) =>
                '<option value="' +
                o +
                '">' +
                o.charAt(0).toUpperCase() +
                o.slice(1) +
                "</option>",
            )
            .join("");
          descSel.value = a.description || "lamb";
        } else {
          descWrap.style.display = "none";
        }
        const gradeWrap = document.getElementById("editAnimalGradeWrap");
        if (a.type === "goat" || a.type === "lamb") {
          gradeWrap.style.display = "";
          document.getElementById("editAnimalGrade").value = a.grade || "";
          _editAnimalGradeState = {
            originalGrade: a.grade || null,
            pendingEntry: null,
          };
        } else {
          gradeWrap.style.display = "none";
          _editAnimalGradeState = null;
        }
        populateProviderSelect(
          document.getElementById("editAnimalProv"),
          a.prov_id,
        );
        document.getElementById("editAnimalKill").value = a.kill_date
          ? a.kill_date.split("T")[0]
          : "";
        document.getElementById("editAnimalProc").value = a.process_date
          ? a.process_date.split("T")[0]
          : "";
      }
      async function saveEditAnimal() {
        const type = document.getElementById("editAnimalType").value,
          sid = parseInt(document.getElementById("editAnimalSerial").value),
          body = {};
        const gradeWrap = document.getElementById("editAnimalGradeWrap");
        const gradeRecord = allGrades.find((g) => String(g.serial_id) === String(sid));
        let newGrade = null;
        let gradeChanged = false;
        const desc = document.getElementById("editAnimalDesc")?.value;
        if (
          desc !== undefined &&
          document.getElementById("editAnimalDescWrap").style.display !== "none"
        )
          body.description = desc || null;
        if (gradeWrap.style.display !== "none") {
          newGrade = document.getElementById("editAnimalGrade").value || null;
          gradeChanged =
            newGrade !== (_editAnimalGradeState?.originalGrade || null);
          if (!gradeRecord) {
            body.grade = newGrade;
          }
        }
        const hw = parseFloat(document.getElementById("editAnimalHW").value);
        if (!isNaN(hw)) body.hang_weight = hw;
        const lw = parseFloat(document.getElementById("editAnimalLW").value);
        if (!isNaN(lw)) body.live_weight = lw;
        const prov = document.getElementById("editAnimalProv").value;
        body.prov_id = prov ? parseInt(prov) : null;
        const kd = document.getElementById("editAnimalKill").value;
        if (kd) body.kill_date = kd;
        const pd = document.getElementById("editAnimalProc").value;
        if (pd) body.process_date = pd;
        try {
          if (gradeChanged && !_editAnimalGradeState?.pendingEntry) {
            throw new Error("Manual grade changes require an annotation");
          }
          const r = await HerdAuth.fetch(
            "/db/" + type + "s/" + sid,
            {
              method: "PUT",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(body),
            },
          );
          if (!r.ok) throw new Error("Update failed: " + r.status);
          if (gradeRecord && gradeChanged) {
            const history = Array.isArray(gradeRecord.manual_override_history)
              ? gradeRecord.manual_override_history.slice()
              : [];
            const gradeResp = await HerdAuth.fetch(
              "/db/grading/result/" + gradeRecord.id,
              {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                  grade: newGrade,
                  manual_override_history: history.concat(
                    _editAnimalGradeState.pendingEntry,
                  ),
                }),
              },
            );
            if (!gradeResp.ok) {
              const err = await gradeResp.json().catch(() => ({}));
              throw new Error(err.detail || "Grade update failed: " + gradeResp.status);
            }
          }
          closeModal("modalEditAnimal");
          _editAnimalGradeState = null;
          if (typeof reloadGradeDataAndUI === "function") {
              await reloadGradeDataAndUI();
          } else {
              if (type === "goat") await loadGoats();
              else if (type === "chicken") await loadChickens();
              else await loadLambs();
              await loadGrades().catch(() => {});
              updateDashboard();
              filterAnimals();
              renderGradeHistory();
          }
          selectAnimal(sid, type);
          showToast("success", "Updated " + type + " #" + sid);
        } catch (err) {
          showToast("error", err.message);
        }
      }

      // DELETE ANIMAL
      function deleteAnimal() {
        const a = allAnimals.find(
          (x) =>
            x.serial_id === selectedAnimalId && x.type === selectedAnimalType,
        );
        if (!a) return;
        const grades = allGrades.filter((g) => g.serial_id == a.serial_id);
        let msg = "Permanently delete " + a.type + " #" + a.serial_id + "?";
        if (grades.length)
          msg +=
            " " +
            grades.length +
            " grade result" +
            (grades.length > 1 ? "s" : "") +
            " will also be deleted.";
        document.getElementById("deleteMsg").textContent = msg;
        openModal("modalDelete");
      }
      async function confirmDelete() {
        closeModal("modalDelete");
        const type = selectedAnimalType,
          sid = selectedAnimalId;
        try {
          const grades = allGrades.filter((g) => g.serial_id == sid);
          const r = await HerdAuth.fetch(
            "/db/" + type + "s/" + sid,
            { method: "DELETE" },
          );
          if (!r.ok) {
            const err = await r.json().catch(() => ({}));
            throw new Error(err.detail || "Delete failed: " + r.status);
          }
          showToast(
            "success",
            "Deleted " +
              type +
              " #" +
              sid +
              (grades.length
                ? " + " +
                  grades.length +
                  " grade" +
                  (grades.length > 1 ? "s" : "")
                : ""),
          );
          selectedAnimalId = null;
          selectedAnimalType = null;
          document.getElementById("animalDetail").style.display = "none";
          await Promise.all([
            loadGoats().catch(() => {}),
            loadChickens().catch(() => {}),
            loadLambs().catch(() => {}),
            loadGrades().catch(() => {}),
          ]);
          filterAnimals();
          updateDashboard();
          renderGradeHistory();
          computeNextGradeId();
        } catch (err) {
          showToast("error", err.message);
        }
      }

      // PROVIDERS
