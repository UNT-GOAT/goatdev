      function renderGradeHistory() {
        const body = document.getElementById("gradeHistoryBody");
        if (!allGrades.length) {
          body.innerHTML =
            '<div class="no-results">No grading results yet</div>';
          return;
        }
        body.innerHTML = allGrades
          .map((g) => {
            const speciesCol = g.species
              ? typeBadge(g.species)
              : '<span style="color:var(--text-faint);font-size:10px">—</span>';
            const descCol = g.description
              ? '<span style="color:var(--text-secondary);font-size:12px">' +
                esc(g.description) +
                "</span>"
              : '<span style="color:var(--text-faint);font-size:12px">—</span>';
            const provName = g.serial_id
              ? (allAnimals.find((a) => a.serial_id == g.serial_id) || {})
                  .providerName || "—"
              : "—";
            return (
              '<div class="tbl-row" style="grid-template-columns:80px 76px 1fr 72px 94px 1fr 90px 50px"><span style="color:var(--text-primary);font-weight:600;font-family:\'JetBrains Mono\',monospace;font-size:12px">' +
              g.serial_id +
              "</span>" +
              speciesCol +
              descCol +
              "<span style=\"color:var(--text-secondary);font-family:'JetBrains Mono',monospace;font-size:12px\">" +
              wt(g.live_weight) +
              "</span>" +
              gradeBadge(g.grade) +
              '<span style="color:var(--text-muted);font-size:12px">' +
              esc(provName) +
              "</span>" +
              '<span style="color:var(--text-faint);font-size:10px">' +
              fmtDate(g.graded_at) +
              "</span>" +
              '<button class="btn btn-sm" onclick="viewGradeDetail(' +
              g.id +
              ')">Detail</button></div>'
            );
          })
          .join("");
      }
      async function viewGradeDetail(resultId) {
        try {
          const r = await dbFetch("/grading/result/" + resultId);
          openGradeReview(
            {
              grade: r.grade,
              analysis_key: r.analysis_key || null,
              live_weight: r.live_weight,
              all_views_ok: r.all_views_ok,
              measurements: r.measurements,
              total_sec: r.total_sec ? parseFloat(r.total_sec) : null,
              warnings: r.warnings,
              species: r.species || null,
              description: r.description || null,
              view_errors: r.view_errors || [],
              grade_details: r.grade_details || null,
              manual_override_history: r.manual_override_history || [],
            },
            r.analysis_key || r.serial_id,
          );
          document.getElementById("reviewFooter").style.display = "none";
          document.getElementById("reviewRetryFooter").style.display = "none";
          document.getElementById("reviewFinalGradeSection").style.display = "none";
          document.getElementById("reviewSubtitle").textContent =
            "Saved grade — click to edit";
          _reviewGradeId = resultId;
          document.getElementById("reviewEditSection").style.display = "";
          document.getElementById("reviewEditGrade").value = r.grade || "";
          _reviewEditGradeState = {
            originalGrade: r.grade || null,
            pendingEntry: null,
          };
        } catch (err) {
          showToast("error", "Could not load grade detail");
        }
      }

      // GRADE EDIT/DELETE — syncs to animal record
      async function saveEditGrade() {
        if (!_reviewGradeId) return;
        const newGrade = document.getElementById("reviewEditGrade").value || null;
        const currentGrade = _reviewCurrentResult?.grade || null;
        const currentHistory = Array.isArray(_reviewCurrentResult?.manual_override_history)
          ? _reviewCurrentResult.manual_override_history.slice()
          : [];
        try {
            const payload = { grade: newGrade };
            if (newGrade !== currentGrade) {
              if (!_reviewEditGradeState?.pendingEntry) {
                throw new Error("Manual grade changes require an annotation");
              }
              payload.manual_override_history = currentHistory.concat(
                _reviewEditGradeState.pendingEntry,
              );
            }
            const r = await HerdAuth.fetch(
                "/db/grading/result/" + _reviewGradeId,
                {
                    method: "PUT",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                },
            );
            if (!r.ok) {
                const err = await r.json().catch(() => ({}));
                throw new Error(err.detail || "Failed: " + r.status);
            }

            closeModal("modalGradeReview");
            _reviewCurrentResult = null;
            _reviewEditGradeState = null;

            const gradeResult = allGrades.find((g) => g.id === _reviewGradeId);
            const serialId = gradeResult ? gradeResult.serial_id : null;
            const animal = serialId
                ? allAnimals.find((a) => String(a.serial_id) === String(serialId))
                : null;

            if (animal) {
                if (typeof reloadGradeDataAndUI === "function") {
                    await reloadGradeDataAndUI();
                } else {
                    await Promise.all([loadGrades().catch(() => {}), loadGoats().catch(() => {}), loadLambs().catch(() => {})]);
                    updateDashboard();
                    filterAnimals();
                    renderGradeHistory();
                }
                selectAnimal(Number(serialId), animal.type);
            } else {
                await loadGrades().catch(() => {});
                renderGradeHistory();
                updateDashboard();
            }

            showToast("success", "Grade updated to " + (newGrade || "Ungraded"));
        } catch (err) {
            showToast("error", err.message);
        }
    }
      async function deleteGradeResult() {
        if (!_reviewGradeId) return;
        if (!confirm("Delete this grade result permanently?")) return;
        try {
          const r = await HerdAuth.fetch(
            "/db/grading/result/" + _reviewGradeId,
            { method: "DELETE" },
          );
          if (!r.ok) {
            const err = await r.json().catch(() => ({}));
            throw new Error(err.detail || "Delete failed: " + r.status);
          }
          closeModal("modalGradeReview");
          _reviewCurrentResult = null;
          showToast("success", "Grade result deleted");
          _reviewGradeId = null;
          await loadGrades().catch(() => {});
          renderGradeHistory();
          updateDashboard();
        } catch (err) {
          showToast("error", err.message);
        }
      }

      // LIGHTBOX
