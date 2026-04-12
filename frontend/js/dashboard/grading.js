
      async function reloadGradeDataAndUI() {
        await Promise.all([
          loadGoats().catch(() => {}),
          loadLambs().catch(() => {}),
          loadChickens().catch(() => {}),
          loadGrades().catch(() => {}),
        ]);
        updateDashboard();
        filterAnimals();
        renderGradeHistory();
      }

      async function syncGradeEverywhere(serialId, type, newGrade) {
        try {
          await HerdAuth.fetch(
            "/db/" + type + "s/" + serialId,
            {
              method: "PUT",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ grade: newGrade }),
            },
          );
        } catch (err) {
          console.warn("Animal grade sync failed:", err);
        }

        const gradeRecord = allGrades.find(
          (g) => String(g.serial_id) === String(serialId),
        );
        if (gradeRecord) {
          try {
            await HerdAuth.fetch(
              "/db/grading/result/" + gradeRecord.id,
              {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ grade: newGrade }),
              },
            );
          } catch (err) {
            console.warn("Grade result sync failed:", err);
          }
        }

        await reloadGradeDataAndUI();
      }

      function _renderOverrideHistory(history) {
        const wrap = document.getElementById("reviewOverrideHistoryWrap");
        const body = document.getElementById("reviewOverrideHistory");
        if (!Array.isArray(history) || !history.length) {
          wrap.style.display = "none";
          body.innerHTML = "";
          return;
        }
        wrap.style.display = "";
        body.innerHTML = history
          .map((entry) => {
            const fromGrade = entry.from_grade || "Ungraded";
            const toGrade = entry.to_grade || "Ungraded";
            const when = entry.changed_at
              ? new Date(entry.changed_at).toLocaleString("en-US", {
                  month: "short",
                  day: "numeric",
                  year: "numeric",
                  hour: "numeric",
                  minute: "2-digit",
                })
              : "—";
            const who = entry.username || "unknown";
            const reason = entry.reason_code ? " · " + esc(entry.reason_code) : "";
            return (
              '<div style="padding:10px 0;border-bottom:1px solid rgba(255,255,255,0.06)">' +
              '<div style="font-weight:600;color:var(--text-primary)">' +
              esc(fromGrade) + " → " + esc(toGrade) +
              "</div>" +
              '<div style="font-size:10px;color:var(--text-faint);margin:3px 0 6px 0">' +
              esc(who) + " · " + esc(when) + reason +
              "</div>" +
              '<div>' + esc(entry.annotation || "") + "</div>" +
              "</div>"
            );
          })
          .join("");
      }

      function _normalizeGradeValue(value) {
        return value || null;
      }

      function _setGradeSelectValue(selectId, value) {
        const el = document.getElementById(selectId);
        if (el) el.value = value || "";
      }

      function _buildOverrideEntry(fromGrade, toGrade, changeContext) {
        return {
          from_grade: fromGrade || null,
          to_grade: toGrade || null,
          annotation: "",
          reason_code: "",
          username: _currentUser?.username || "unknown",
          changed_at: new Date().toISOString(),
          change_context: changeContext,
        };
      }

      function _confirmedGradeForState(state) {
        if (!state) return null;
        return _normalizeGradeValue(
          state.pendingEntry ? state.pendingEntry.to_grade : state.originalGrade,
        );
      }

      function _queueManualGradeOverride(stateKey, selectId, nextGrade, changeContext) {
        const state = window[stateKey];
        if (!state) return;

        const desiredGrade = _normalizeGradeValue(nextGrade);
        const originalGrade = _normalizeGradeValue(state.originalGrade);
        const confirmedGrade = _confirmedGradeForState(state);

        if (desiredGrade === confirmedGrade) return;

        if (desiredGrade === originalGrade) {
          state.pendingEntry = null;
          _setGradeSelectValue(selectId, originalGrade);
          return;
        }

        openGradeAnnotationModal({
          onSubmit: async (note, reason) => {
            state.pendingEntry = {
              ..._buildOverrideEntry(originalGrade, desiredGrade, changeContext),
              annotation: note,
              reason_code: reason || null,
            };
            _setGradeSelectValue(selectId, desiredGrade);
          },
          onCancel: () => {
            _setGradeSelectValue(selectId, confirmedGrade);
          },
        });
      }

      function onEditAnimalGradeChange() {
        const select = document.getElementById("editAnimalGrade");
        _queueManualGradeOverride(
          "_editAnimalGradeState",
          "editAnimalGrade",
          select ? select.value : "",
          "saved_result_edit",
        );
      }

      function onReviewEditGradeChange() {
        const select = document.getElementById("reviewEditGrade");
        _queueManualGradeOverride(
          "_reviewEditGradeState",
          "reviewEditGrade",
          select ? select.value : "",
          "saved_result_edit",
        );
      }

      function openGradeAnnotationModal(context) {
        _gradeAnnotationContext = context;
        document.getElementById("gradeAnnotationReason").value = "";
        document.getElementById("gradeAnnotationText").value = "";
        openModal("modalGradeAnnotation");
        setTimeout(() => {
          document.getElementById("gradeAnnotationText").focus();
        }, 50);
      }

      function closeGradeAnnotationModal() {
        const ctx = _gradeAnnotationContext;
        _gradeAnnotationContext = null;
        const modal = document.getElementById("modalGradeAnnotation");
        if (modal) {
          modal.classList.remove("open");
          modal.classList.remove("stacked");
        }
        if (ctx && typeof ctx.onCancel === "function") {
          ctx.onCancel();
        }
      }

      async function submitGradeAnnotation() {
        if (!_gradeAnnotationContext) return;
        const note = (document.getElementById("gradeAnnotationText").value || "").trim();
        const reason = document.getElementById("gradeAnnotationReason").value || "";
        if (!note) {
          showToast("error", "Annotation is required for manual grade changes");
          return;
        }
        const ctx = _gradeAnnotationContext;
        _gradeAnnotationContext = null;
        const modal = document.getElementById("modalGradeAnnotation");
        if (modal) {
          modal.classList.remove("open");
          modal.classList.remove("stacked");
        }
        try {
          await ctx.onSubmit(note, reason);
        } catch (err) {
          showToast("error", err.message || "Could not save grade annotation");
        }
      }

      async function _savePendingGradeResult() {
        const btn = document.getElementById("acceptGradeBtn");
        const species = pendingGradeResult.species;
        const endpoint = "/" + species + "s";
        const label = species.charAt(0).toUpperCase() + species.slice(1);
        const isExisting = !!pendingGradeResult._isExisting;
        const gradeProv = document.getElementById("gradeProv").value;
        const gradeKillDate = document.getElementById("gradeKillDate").value;

        try {
          if (isExisting) {
            gradeLog(
              "Updating existing " +
                species +
                " #" +
                pendingGradeResult.serial_id +
                "...",
              "info",
            );
            const putBody = {
              grade: pendingGradeResult.grade || null,
              live_weight: pendingGradeResult.live_weight,
            };
            if (gradeProv) putBody.prov_id = parseInt(gradeProv);
            if (gradeKillDate) putBody.kill_date = gradeKillDate;
            const putResp = await HerdAuth.fetch(
              "/db" + endpoint + "/" + pendingGradeResult.serial_id,
              {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(putBody),
              },
            );
            if (!putResp.ok) {
              const err = await putResp.json().catch(() => ({}));
              throw new Error(
                "Update failed: " + (err.detail || putResp.status),
              );
            }
            gradeLog(
              label + " #" + pendingGradeResult.serial_id + " updated",
              "ok",
            );
          } else if (!_gradeAnimalCreated) {
            gradeLog(
              "Creating " +
                species +
                " #" +
                pendingGradeResult.serial_id +
                "...",
              "info",
            );
            const createBody = {
              serial_id: pendingGradeResult.serial_id,
              live_weight: pendingGradeResult.live_weight,
              grade: pendingGradeResult.grade || null,
              description: pendingGradeResult.description || null,
            };
            if (gradeProv) createBody.prov_id = parseInt(gradeProv);
            if (gradeKillDate) createBody.kill_date = gradeKillDate;
            const animalResp = await HerdAuth.fetch("/db" + endpoint, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(createBody),
            });
            if (!animalResp.ok) {
              const err = await animalResp.json().catch(() => ({}));
              throw new Error(
                label +
                  " creation failed: " +
                  (err.detail || animalResp.status),
              );
            }
            const created = await animalResp.json();
            pendingGradeResult.serial_id = created.serial_id;
            _gradeAnimalCreated = true;
            gradeLog(label + " #" + created.serial_id + " created", "ok");
          }
          gradeLog("Saving grade result...", "info");
          const gradePayload = { ...pendingGradeResult };
          delete gradePayload.species;
          delete gradePayload.description;
          delete gradePayload._isExisting;
          delete gradePayload.view_errors;
          delete gradePayload._aiGrade;
          const gradeResp = await HerdAuth.fetch("/db/grading", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(gradePayload),
          });
          if (!gradeResp.ok) {
            const err = await gradeResp.json().catch(() => ({}));
            gradeLog("Grade save failed — click Retry.", "err");
            showToast("error", err.detail || "Grade save failed — retry or discard");
            btn.disabled = false;
            btn.textContent = "Retry Save";
            return;
          }
          _gradeAnimalCreated = false;
          _gradeExistingId = null;
          _gradeReservedId = null;
          closeModal("modalGradeReview");
          _reviewCurrentResult = null;
          showToast(
            "success",
            label +
              " #" +
              pendingGradeResult.serial_id +
              " " +
              (isExisting ? "graded" : "created") +
              " — " +
              (pendingGradeResult.grade || "Ungraded"),
          );
          gradeLog("Saved to database", "ok");
          const _savedSerial = pendingGradeResult?.serial_id;
          const _savedSpecies = pendingGradeResult?.species;
          pendingGradeResult = null;
          _editAnimalGradeState = null;
          _reviewEditGradeState = null;
          await reloadGradeDataAndUI();
          computeNextGradeId();
          if (_savedSerial && _savedSpecies) {
              selectAnimal(Number(_savedSerial), _savedSpecies);
          }
        } catch (err) {
          showToast("error", err.message);
          gradeLog("Error: " + err.message, "err");
        } finally {
          if (!_gradeAnimalCreated) {
            btn.disabled = false;
            btn.textContent = "Accept & Save";
          }
        }
      }

      function onGradeSpeciesChange() {
        const species = document.getElementById("gradeSpecies").value;
        const descEl = document.getElementById("gradeDesc");
        const opts = DESCRIPTION_OPTIONS[species] || [];
        descEl.innerHTML = opts
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
        if (!_gradeExistingId) computeNextGradeId();
      }
      async function computeNextGradeId() {
        _gradeExistingId = null;
        _gradeReservedId = null;
        const hint = document.getElementById("gradeSerialHint");
        const input = document.getElementById("gradeSerial");
        const btn = document.getElementById("gradeBtn");
        hint.textContent = "Fetching next ID...";
        hint.className = "field-hint";
        input.value = "";
        btn.disabled = true;
        nextGradeId = await fetchNextGlobalId();
        input.value = nextGradeId;
        hint.textContent = "";
        hint.className = "field-hint valid";
        btn.disabled = !piConnected;
        populateProviderSelect(document.getElementById("gradeProv"));
        if (!document.getElementById("gradeKillDate").value)
          document.getElementById("gradeKillDate").value = todayISO();
      }
      async function resolveGradeSerialId() {
        if (_gradeExistingId) return _gradeExistingId;
        if (_gradeReservedId) {
          nextGradeId = _gradeReservedId;
          document.getElementById("gradeSerial").value = _gradeReservedId;
          document.getElementById("gradeSerialHint").textContent =
            "Reserved for this grading session";
          document.getElementById("gradeSerialHint").className =
            "field-hint valid";
          return _gradeReservedId;
        }

        const hint = document.getElementById("gradeSerialHint");
        hint.textContent = "Reserving serial ID...";
        hint.className = "field-hint";
        const reservedId = await allocateNextGlobalId();
        _gradeReservedId = reservedId;
        nextGradeId = reservedId;
        document.getElementById("gradeSerial").value = reservedId;
        hint.textContent = "Reserved for this grading session";
        hint.className = "field-hint valid";
        return reservedId;
      }
      function gradeLog(msg, type) {
        const el = document.getElementById("gradeLog");
        el.style.display = "";
        const div = document.createElement("div");
        div.className = type || "info";
        div.textContent = new Date().toLocaleTimeString() + "  " + msg;
        el.appendChild(div);
        el.scrollTop = el.scrollHeight;
      }
      async function startGrade() {
        if (!piConnected) {
          showToast("error", "Grader is offline");
          gradeLog("Grader is offline.", "err");
          return;
        }
        let serialId = _gradeExistingId || nextGradeId;
        const species = document.getElementById("gradeSpecies").value;
        const desc = document.getElementById("gradeDesc").value;
        const lw = parseFloat(document.getElementById("gradeWeight").value);
        if (!desc) {
          showToast("error", "Select a description");
          return;
        }
        if (!lw || lw <= 0) {
          showToast("error", "Enter a valid live weight");
          return;
        }
        const btn = document.getElementById("gradeBtn");
        btn.disabled = true;
        document.getElementById("gradeLog").innerHTML = "";
        document.getElementById("gradeLog").style.display = "none";
        const isExisting = !!_gradeExistingId;
        try {
          serialId = await resolveGradeSerialId();
          if (!serialId) {
            showToast("error", "No serial ID");
            btn.disabled = false;
            return;
          }
          if (!isExisting) {
            gradeLog("Reserved serial #" + serialId, "ok");
          }
          gradeLog(
            "Starting grade for " +
              (isExisting ? "existing " : "new ") +
              species +
              " (serial " +
              serialId +
              ", " +
              desc +
              ") @ " +
              lw +
              " lbs",
          );
          gradeLog("Verifying grader connection...", "info");
          let pingOk = false;
          try {
            const controller = new AbortController();
            const timer = setTimeout(() => controller.abort(), 5000);
            await HerdAuth.fetch("/api/prod/status", {
              signal: controller.signal,
            });
            clearTimeout(timer);
            pingOk = true;
          } catch (e) {}
          if (!pingOk) {
            gradeLog("Grader is unreachable.", "err");
            piConnected = false;
            document.getElementById("piStatus").innerHTML =
              '<span class="dot red"></span><span>Grader offline</span>';
            showToast("error", "Grader unreachable");
            btn.disabled = false;
            return;
          }
          gradeLog("POST /api/prod/grade ...", "info");
          const resp = await HerdAuth.fetch(
            "/api/prod/grade",
            {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                serial_id: String(serialId),
                live_weight: lw,
                description: desc,
              }),
            },
          );
          if (!resp.ok) {
            const body = await resp.json().catch(() => ({}));
            if (body.error_code === "CAMERA_BELOW_OPERATING_TEMP") {
              const waitMsg = body.message || CAMERA_COLD_MESSAGE;
              cameraColdBlocked = true;
              cameraColdMessage = waitMsg;
              cameraSafety = {
                cold_gate_active: true,
                blocked_cameras: body.blocked_cameras || [],
                temps_f: body.temps_f || null,
                min_operating_temp_f: body.min_operating_temp_f || 32,
                message: waitMsg,
              };
              gradeLog("Rejected: " + waitMsg, "err");
              showToast("error", waitMsg);
              if (currentPage === "grading" && camerasActive) startCameraStreams();
              btn.disabled = false;
              return;
            }
            gradeLog(
              "Rejected: " +
                (body.message ||
                  body.error_code ||
                  body.detail ||
                  "HTTP " + resp.status),
              "err",
            );
            if (body.fix) gradeLog("Fix: " + body.fix, "warn");
            btn.disabled = false;
            return;
          }
          gradeLog("Capturing images...", "ok");
          const result = await pollGradeStatus();
          if (result && result.success !== false && !result.error) {
            gradeLog("Complete — opening review...", "ok");
            pendingGradeResult = {
              serial_id: serialId,
              species,
              description: desc,
              grade: result.grade || null,
              _aiGrade: result.grade || null,
              live_weight: lw,
              all_views_ok:
                result.all_views_successful ?? result.all_views_ok ?? null,
              measurements: result.measurements || null,
              capture_sec: result.timing?.capture_sec || null,
              ec2_sec: result.timing?.ec2_sec || null,
              total_sec: result.timing?.total_sec || null,
              warnings: result.warnings || [],
              view_errors: result.view_errors || [],
              grade_details: result.grade_details || null,
              manual_override_history:
                (
                  allGrades.find((g) => String(g.serial_id) === String(serialId)) || {}
                ).manual_override_history || [],
              _isExisting: isExisting,
            };
            openGradeReview(pendingGradeResult, serialId);
          } else {
            gradeLog("Failed: " + (result?.error || "unknown"), "err");
            if (result?.fix) gradeLog("Fix: " + result.fix, "warn");
          }
        } catch (err) {
          gradeLog("Network error: " + err.message, "err");
          piConnected = false;
          document.getElementById("piStatus").innerHTML =
            '<span class="dot red"></span><span>Grader offline</span>';
        } finally {
          btn.disabled = false;
        }
      }
      async function pollGradeStatus() {
        let lastProgress = null;
        for (let i = 0; i < 240; i++) {
          await new Promise((r) => setTimeout(r, 500));
          let data;
          try {
            const r = await HerdAuth.fetch(
              "/api/prod/status",
            );
            data = await r.json();
          } catch (e) {
            gradeLog("Retrying...", "warn");
            continue;
          }
          if (data.progress && data.progress !== lastProgress) {
            lastProgress = data.progress;
            const msgs = {
              starting: "Initializing...",
              capturing: "Capturing images...",
              grading: "Processing on EC2...",
              archiving: "Archiving to S3...",
            };
            gradeLog(msgs[data.progress] || data.progress, "info");
          }
          if (!data.active)
            return data.last_error
              ? { success: false, error: data.last_error }
              : data.last_result || { success: false, error: "No result" };
        }
        return { success: false, error: "Timed out" };
      }

      // GRADE REVIEW — with failure/retry
      function openGradeReview(result, serialId) {
        openModal("modalGradeReview");
        document.getElementById("reviewSubtitle").textContent =
          "Review before saving";
        document.getElementById("reviewEditSection").style.display = "none";
        document.getElementById("reviewFinalGradeSection").style.display = "none";
        _reviewGradeId = null;
        _reviewCurrentResult = result;
        _reviewEditGradeState = null;
        const isOk = result.all_views_ok;
        const badgeEl = document.getElementById("reviewGradeBadge");
        badgeEl.style.color = "";
        badgeEl.style.background = "";
        if (isOk && result.grade) {
          badgeEl.textContent = result.grade;
          badgeEl.className =
            "preview-grade-badge badge lg " + gradeClass(result.grade);
        } else if (!isOk && !_devForceSave) {
          badgeEl.textContent = "FAILED";
          badgeEl.className = "preview-grade-badge badge lg";
          badgeEl.style.color = "var(--red)";
          badgeEl.style.background = "rgba(229,57,53,0.08)";
        } else {
          badgeEl.textContent = "UNGRADED";
          badgeEl.className = "preview-grade-badge badge lg badge-ungraded";
        }
        badgeEl.style.fontSize = "24px";
        document.getElementById("reviewWeight").textContent = result.live_weight
          ? result.live_weight + " lbs"
          : "—";
        document.getElementById("reviewSpecies").textContent = result.species
          ? result.species.charAt(0).toUpperCase() +
            result.species.slice(1) +
            (result.description ? " / " + result.description : "")
          : "—";
        document.getElementById("reviewViews").textContent = isOk
          ? "3/3"
          : "Failed";
        document.getElementById("reviewViews").style.color = isOk
          ? ""
          : "var(--red)";
        document.getElementById("reviewTime").textContent = result.total_sec
          ? parseFloat(result.total_sec).toFixed(1) + "s"
          : "—";
        ["side", "top", "front"].forEach((view) => {
          const slot = document.getElementById(
            "review" + view.charAt(0).toUpperCase() + view.slice(1),
          );
          const existing = slot.querySelector("img");
          if (existing) existing.remove();
          const img = document.createElement("img");
          img.alt = view + " debug";
          img.onerror = function () {
            this.style.opacity = ".2";
          };
          slot.insertBefore(img, slot.firstChild);
          HerdAuth.setPiImageSource(img, {
            kind: "debug",
            view,
            serialId: String(serialId),
          }).catch(() => {
            img.style.opacity = ".2";
          });
        });
        document.getElementById("reviewMeasurements").textContent =
          result.measurements
            ? JSON.stringify(result.measurements, null, 2)
            : "None";
        const ww = document.getElementById("reviewWarningsWrap"),
          wl = document.getElementById("reviewWarningsList");
        if (result.warnings && result.warnings.length) {
          ww.style.display = "";
          wl.innerHTML = result.warnings
            .map((w) => "<li>" + esc(w) + "</li>")
            .join("");
        } else {
          ww.style.display = "none";
        }
        const failureEl = document.getElementById("reviewFailure");
        const reasonsEl = document.getElementById("reviewFailureReasons");
        const okFooter = document.getElementById("reviewFooter");
        const retryFooter = document.getElementById("reviewRetryFooter");
        if (!isOk) {
          const reasons = [];
          if (result.view_errors && result.view_errors.length) {
            result.view_errors.forEach((ve) => {
              const vn =
                (ve.view || "").charAt(0).toUpperCase() +
                (ve.view || "").slice(1);
              reasons.push(
                "<strong>" + vn + " view:</strong> " + esc(ve.error || ve),
              );
            });
          }
          if (result.warnings && result.warnings.length) {
            result.warnings.forEach((w) => {
              if (w.toLowerCase().includes("confidence")) reasons.push(esc(w));
            });
          }
          if (reasons.length === 0)
            reasons.push(
              "One or more camera views did not produce usable data.",
            );
          reasonsEl.innerHTML = reasons
            .map((r) => "<li>" + r + "</li>")
            .join("");
          failureEl.style.display = "";
          okFooter.style.display = "none";
          retryFooter.style.display = "";
          document.getElementById("reviewSubtitle").textContent =
            "Grade failed — review and retry";
        } else {
          failureEl.style.display = "none";
          okFooter.style.display = "";
          retryFooter.style.display = "none";
        }
        // Grade details / reasoning
        const detailsEl = document.getElementById("reviewGradeDetails");
        const gd = result.grade_details;
        if (gd) {
          detailsEl.style.display = "";
          document.getElementById("reviewCI").textContent =
            gd.ci != null ? gd.ci.toFixed(3) : "\u2014";
          const selectTier = gd.tier_comparison
            ? gd.tier_comparison[gd.tier_comparison.length - 1]
            : null;
          document.getElementById("reviewCI").style.color =
            selectTier && gd.ci != null
              ? (selectTier.ci_ok ? "" : "var(--red)")
              : "";
          document.getElementById("reviewMDR").textContent =
            gd.mdr != null ? gd.mdr.toFixed(3) : "\u2014";
          document.getElementById("reviewMDR").style.color =
            selectTier && gd.mdr != null
              ? (selectTier.mdr_ok ? "" : "var(--red)")
              : "";
          document.getElementById("reviewCategory").textContent =
            gd.category || "\u2014";
          document.getElementById("reviewReason").textContent = gd.reason || "";
          if (gd.tier_comparison && gd.tier_comparison.length) {
            const chk = (v) => v === true ? "\u2713" : v === false ? "\u2717" : "\u2014";
            const dlt = (v) => v != null ? (v >= 0 ? "+" : "") + v.toFixed(3) : "";
            let tbl =
              '<table style="width:100%;border-collapse:collapse;text-align:center">' +
              "<tr>" +
              '<th style="text-align:left;padding:3px 8px 3px 0">Tier</th>' +
              '<th style="padding:3px 6px">Weight</th>' +
              '<th style="padding:3px 6px">CI</th>' +
              '<th style="padding:3px 6px">MDR</th>' +
              "</tr>";
            gd.tier_comparison.forEach((t) => {
              const allOk =
                t.weight_ok &&
                (t.ci_ok === true || t.ci_ok === null) &&
                (t.mdr_ok === true || t.mdr_ok === null);
              const rc = allOk ? "color:var(--accent)" : "";
              tbl +=
                '<tr style="' + rc + '">' +
                '<td style="text-align:left;padding:3px 8px 3px 0;font-weight:500">' +
                esc(t.tier) + "</td>" +
                '<td style="padding:3px 6px">' +
                chk(t.weight_ok) + " " + t.min_weight + "</td>" +
                '<td style="padding:3px 6px">' +
                chk(t.ci_ok) + " " + t.min_ci +
                (t.ci_delta != null
                  ? ' <span style="opacity:.6">(' + dlt(t.ci_delta) + ")</span>"
                  : "") +
                "</td>" +
                '<td style="padding:3px 6px">' +
                chk(t.mdr_ok) + " " + t.min_mdr +
                (t.mdr_delta != null
                  ? ' <span style="opacity:.6">(' + dlt(t.mdr_delta) + ")</span>"
                  : "") +
                "</td></tr>";
            });
            tbl += "</table>";
            document.getElementById("reviewTierTable").innerHTML = tbl;
          } else {
            document.getElementById("reviewTierTable").innerHTML = "";
          }
        } else {
          detailsEl.style.display = "none";
        }
        _renderOverrideHistory(result.manual_override_history || []);
      }
      function retryGrade() {
        closeModal("modalGradeReview");
        pendingGradeResult = null;
        _reviewCurrentResult = null;
        _gradeAnimalCreated = false;
        gradeLog("Retrying — previous attempt had view failures", "warn");
        document.getElementById("gradeWeight").focus();
      }
      function confirmDiscard() {
        closeModal("modalDiscard");
        closeModal("modalGradeReview");
        pendingGradeResult = null;
        _reviewCurrentResult = null;
        _gradeAnimalCreated = false;
        _gradeExistingId = null;
        _gradeReservedId = null;
        gradeLog("Discarded by operator", "warn");
        showToast("success", "Grade discarded");
        document.getElementById("gradeWeight").value = "";
        document.getElementById("gradeProv").value = "";
        document.getElementById("gradeKillDate").value = "";
        document.getElementById("gradeLog").innerHTML = "";
        document.getElementById("gradeLog").style.display = "none";
        computeNextGradeId();
      }
      function rejectGrade() {
        openModal("modalDiscard");
      }

      async function acceptGrade() {
        if (!pendingGradeResult) return;
        const btn = document.getElementById("acceptGradeBtn");
        btn.disabled = true;
        btn.textContent = "Saving...";
        await _savePendingGradeResult();
      }

      // GRADE HISTORY

      // DEV-ONLY: Manual Image Upload Grading
      // ============================================================
      const _devFiles = { side: null, top: null, front: null };
      function _isDevUser() {
        return _currentUser && _currentUser.username === "dev";
      }
      function _showDevUploadIfDev() {
        const el = document.getElementById("devUploadSection");
        if (el) el.style.display = _isDevUser() ? "" : "none";
      }
      function handleDevUpload(view, input) {
        const file = input.files[0];
        if (!file) return;
        _devFiles[view] = file;
        const drop = document.getElementById("devDrop-" + view);
        const existing = drop.querySelector("img");
        if (existing) existing.remove();
        const img = document.createElement("img");
        img.src = URL.createObjectURL(file);
        img.style.cssText =
          "width:100%;height:100%;object-fit:contain;display:block";
        drop.insertBefore(img, drop.firstChild);
        drop.style.borderColor = "var(--accent)";
        drop.style.borderStyle = "solid";
        _updateDevStatus();
      }
      function _updateDevStatus() {
        const count = Object.values(_devFiles).filter(Boolean).length;
        document.getElementById("devUploadStatus").textContent =
          count + "/3 images";
        document.getElementById("devUploadStatus").style.color =
          count === 3 ? "var(--accent)" : "var(--text-faint)";
        document.getElementById("devGradeBtn").disabled = count < 3;
      }
      async function startDevGrade() {
        let serialId = _gradeExistingId || nextGradeId;
        const lw = parseFloat(document.getElementById("gradeWeight").value);
        if (!lw || lw <= 0) {
          showToast("error", "Enter a valid live weight");
          return;
        }
        const desc = document.getElementById("gradeDesc").value;
        if (!desc) {
          showToast("error", "Select a description");
          return;
        }
        const species = document.getElementById("gradeSpecies").value;
        const btn = document.getElementById("devGradeBtn");
        btn.disabled = true;
        document.getElementById("gradeLog").innerHTML = "";
        document.getElementById("gradeLog").style.display = "none";
        const isExisting = !!_gradeExistingId;
        try {
          serialId = await resolveGradeSerialId();
          if (!serialId) {
            showToast("error", "No serial ID");
            btn.disabled = false;
            return;
          }
          if (!isExisting) {
            gradeLog("Reserved serial #" + serialId, "ok");
          }
          gradeLog(
            "DEV: Upload grade for " +
              species +
              " #" +
              serialId +
              " @ " +
              lw +
              " lbs",
            "warn",
          );
          gradeLog("Uploading 3 images directly to Pi → EC2...", "info");
          const formData = new FormData();
          formData.append("serial_id", String(serialId));
          formData.append("live_weight", lw);
          formData.append("side_image", _devFiles.side);
          formData.append("top_image", _devFiles.top);
          formData.append("front_image", _devFiles.front);
          const resp = await HerdAuth.fetch(
            "/api/prod/grade/test",
            { method: "POST", body: formData },
          );
          const body = await resp.json();
          if (!resp.ok) {
            gradeLog(
              "Upload grade rejected: " +
                (body.message || body.error_code || resp.status),
              "err",
            );
            if (body.fix) gradeLog("Fix: " + body.fix, "warn");
            btn.disabled = false;
            return;
          }
          gradeLog("Grade returned — opening review...", "ok");
          pendingGradeResult = {
            serial_id: serialId,
            species,
            description: desc,
            grade: body.grade || null,
            live_weight: lw,
            all_views_ok:
              body.all_views_successful ?? body.all_views_ok ?? null,
            measurements: body.measurements || null,
            capture_sec: body.timing?.capture_sec || null,
            ec2_sec: body.timing?.ec2_sec || null,
            total_sec: body.timing?.total_sec || null,
            warnings: body.warnings || [],
            view_errors: body.view_errors || [],
            grade_details: body.grade_details || null,
            _isExisting: isExisting,
          };
          openGradeReview(pendingGradeResult, serialId);
        } catch (err) {
          gradeLog("Network error: " + err.message, "err");
        } finally {
          btn.disabled = false;
        }
      }

      // ============================================================
