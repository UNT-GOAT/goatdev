      // ============================================================

      function startCaptureCams() {
        if (captureCamsActive) return;
        captureCamsActive = true;

        const container = document.getElementById("captureCams");
        const cams = ["side", "top", "front"];
        const slots = container.querySelectorAll(".cam-slot");

        slots.forEach((slot, i) => {
          const cam = cams[i];
          const offline = slot.querySelector(".cam-offline");

          let img = slot.querySelector("img[data-cam]");
          if (img) {
            clearTimeout(img._retryTimer);
            img.src = "";
            img.remove();
          }

          img = document.createElement("img");
          img.dataset.cam = cam;
          img.alt = cam;
          img.onload = function () {
            if (offline) offline.style.display = "none";
          };
          img.onerror = function () {
            this.style.display = "none";
            if (offline) {
              offline.textContent = piConnected
                ? "Stream interrupted — retrying..."
                : "Waiting for grader...";
              offline.style.display = "";
            }
            clearTimeout(this._retryTimer);
            this._retryTimer = setTimeout(() => {
              if (
                currentPage === "capture" &&
                piConnected &&
                captureCamsActive
              ) {
                this.style.display = "";
                HerdAuth.setPiImageSource(this, {
                  kind: "stream",
                  view: cam,
                }).catch(() => {
                  if (typeof this.onerror === "function") this.onerror.call(this);
                });
              }
            }, 10000);
          };
          slot.insertBefore(img, slot.firstChild);
          HerdAuth.setPiImageSource(img, {
            kind: "stream",
            view: cam,
          }).catch(() => {
            if (typeof img.onerror === "function") img.onerror.call(img);
          });
        });
      }

      function stopCaptureCams() {
        if (!captureCamsActive) return;
        captureCamsActive = false;

        const container = document.getElementById("captureCams");
        container.querySelectorAll("img[data-cam]").forEach((img) => {
          clearTimeout(img._retryTimer);
          img.src = "";
          img.remove();
        });
        container.querySelectorAll(".cam-offline").forEach((el) => {
          el.style.display = "";
          el.textContent = "Waiting for grader...";
        });
      }

      // Refresh capture cam tokens periodically
      function refreshCaptureCamTokens() {
        if (!captureCamsActive || currentPage !== "capture") return;
      }
      setInterval(refreshCaptureCamTokens, 10 * 60 * 1000);

      // ============================================================
      // HEATERS (full override on/auto/off control)

      // CAPTURE (Training Data)
      // ============================================================
      function captureLog(msg, type) {
        const el = document.getElementById("captureLog");
        const div = document.createElement("div");
        div.className = type || "info";
        div.textContent = new Date().toLocaleTimeString() + "  " + msg;
        el.appendChild(div);
        el.scrollTop = el.scrollHeight;
      }

      async function fetchCaptureCount() {
        const counter = document.getElementById("captureCounter");
        counter.textContent = "...";
        counter.style.color = "var(--text-faint)";

        try {
          // Call your new Pi endpoint instead of hitting S3 directly
          const r = await HerdAuth.fetch("/api/capture/next_id");
          if (!r.ok) throw new Error("Server error");

          const data = await r.json();
          captureGoatId = data.next_id;
        } catch (e) {
          console.error(
            "Failed to fetch next_id, falling back to localStorage:",
            e,
          );
          captureGoatId = parseInt(
            localStorage.getItem("captureGoatId") || "1",
          );
        }

        counter.textContent = captureGoatId;
        counter.style.color = "var(--accent)";
        localStorage.setItem("captureGoatId", captureGoatId.toString());
      }

      async function startCapture(isMock) {
        if (isCapturing) return;

        // Check Pi connectivity first — fast-fail instead of hanging
        if (!piConnected) {
          captureLog("ERROR: Pi is offline — cannot capture", "err");
          captureLog(
            "Check that the Pi is powered on and connected to the network",
            "warn",
          );
          showToast("error", "Pi is offline — cannot capture");
          return;
        }

        const btn = isMock
          ? document.getElementById("captureMockBtn")
          : document.getElementById("captureBtn");
        const other = isMock
          ? document.getElementById("captureBtn")
          : document.getElementById("captureMockBtn");
        const status = document.getElementById("captureStatus");
        isCapturing = true;
        btn.disabled = true;
        other.disabled = true;
        btn.classList.add("recording");
        document.getElementById("captureLog").innerHTML = "";
        if (isMock) {
          btn.innerHTML = "TESTING...";
          status.textContent = "Test capture...";
          captureLog("Starting TEST capture (will not save)", "warn");
        } else {
          btn.textContent = "CAPTURING...";
          status.textContent = "Capturing goat #" + captureGoatId + "...";
          captureLog("Starting capture for goat #" + captureGoatId, "info");
        }
        try {
          // Verify Pi is reachable right now
          captureLog("Verifying Pi connection...", "info");
          try {
            const controller = new AbortController();
            const timer = setTimeout(() => controller.abort(), 5000);
            const pingResp = await HerdAuth.fetch("/api/prod/status", {
              signal: controller.signal,
            });
            clearTimeout(timer);
            if (!pingResp.ok) throw new Error("Status " + pingResp.status);
            captureLog("Pi connected", "ok");
          } catch (pingErr) {
            piConnected = false;
            document.getElementById("piStatus").innerHTML =
              '<span class="dot red"></span><span>Pi offline</span>';
            throw new Error("Pi unreachable: " + pingErr.message);
          }

          const goatData = isMock
            ? {}
            : {
                description:
                  document.getElementById("captureDesc").value || null,
                live_weight: document.getElementById("captureLW").value || null,
                grade: document.getElementById("captureGrade").value || null,
              };
          const captureId = isMock ? "test_" + Date.now() : captureGoatId;

          captureLog("POST /api/capture/record ...", "info");
          const response = await HerdAuth.fetch("/api/capture/record", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              goat_id: captureId,
              goat_data: goatData,
              is_test: isMock,
              require_all_cameras: !isMock,
            }),
          });
          if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            const msg =
              err.message || err.error_code || "HTTP " + response.status;
            captureLog("Rejected: " + msg, "err");
            if (err.fix) captureLog("Fix: " + err.fix, "warn");
            if (err.cameras_missing)
              captureLog(
                "Missing cameras: " + err.cameras_missing.join(", "),
                "warn",
              );
            if (err.cameras) {
              Object.entries(err.cameras).forEach(([cam, info]) => {
                if (!info.connected)
                  captureLog("  " + cam + ": not connected", "err");
              });
            }
            throw new Error(msg);
          }
          const startData = await response.json();
          captureLog("Capture started", "ok");
          if (startData.cameras_available?.length > 0)
            captureLog(
              "Cameras: " + startData.cameras_available.join(", "),
              "ok",
            );
          if (startData.warning) captureLog(startData.warning, "warn");
          if (startData.estimated_duration_sec) {
            captureLog(
              "Capturing full-res images (~" +
                startData.estimated_duration_sec +
                "s)...",
              "info",
            );
          } else {
            captureLog("Capturing full-res images...", "info");
          }
          await pollCaptureStatus();
          if (isMock) {
            captureLog("Test complete", "ok");
            status.textContent = "Test complete";
            status.style.color = "var(--accent)";
          } else {
            captureLog("Goat #" + captureGoatId + " saved to S3", "ok");
            status.textContent = "Goat #" + captureGoatId + " captured";
            status.style.color = "var(--accent)";
            captureGoatId++;
            document.getElementById("captureCounter").textContent =
              captureGoatId;
            localStorage.setItem("captureGoatId", captureGoatId.toString());
            document.getElementById("captureDesc").value = "";
            document.getElementById("captureLW").value = "";
            document.getElementById("captureGrade").value = "";
          }
        } catch (err) {
          captureLog("ERROR: " + err.message, "err");
          status.textContent = "Failed";
          status.style.color = "var(--red)";
        } finally {
          isCapturing = false;
          btn.disabled = false;
          other.disabled = false;
          btn.classList.remove("recording");
          if (isMock) btn.innerHTML = "TEST<br>(no save)";
          else btn.textContent = "CAPTURE";
        }
      }

      async function pollCaptureStatus() {
        let lastProgress = null;
        let lastCamera = null;
        for (let i = 0; i < 480; i++) {
          await new Promise((r) => setTimeout(r, 500));
          let data;
          try {
            const r = await HerdAuth.fetch("/api/capture/status");
            data = await r.json();
          } catch (e) {
            captureLog("Retrying...", "warn");
            continue;
          }
          if (data.progress && data.progress !== lastProgress) {
            lastProgress = data.progress;
            const msgs = {
              starting: "Initializing cameras...",
              capturing: "Capturing images...",
              checking_s3: "Checking S3...",
              uploading: "Uploading to cloud...",
            };
            captureLog(msgs[data.progress] || data.progress, "info");
          }
          if (data.progress === "capturing" && data.current_camera) {
            if (data.current_camera !== lastCamera) {
              lastCamera = data.current_camera;
              captureLog(
                "Capturing " + data.current_camera + " camera at full res...",
                "info",
              );
            }
          } else if (data.current_camera == null) {
            lastCamera = null;
          }
          if (!data.active) {
            if (data.last_error) throw new Error(data.last_error);
            if (data.last_result) {
              if (data.last_result.camera_status) {
                for (const [cam, s] of Object.entries(
                  data.last_result.camera_status,
                )) {
                  if (s.startsWith("captured"))
                    captureLog("✓ " + cam + ": " + s, "ok");
                  else if (s === "failed")
                    captureLog("✗ " + cam + ": failed", "err");
                  else captureLog("— " + cam + ": " + s, "warn");
                }
              }
              if (data.last_result.total_images)
                captureLog(
                  "Total images: " + data.last_result.total_images,
                  "info",
                );
              if (!data.last_result.success) throw new Error("Capture failed");
            }
            return;
          }
        }
        throw new Error("Timed out");
      }

      // ============================================================
      // BOOT
