      function getCameraBlockedMessage() {
        return cameraColdMessage || CAMERA_COLD_MESSAGE;
      }
      function setCameraBlockedState(slot, msg) {
        const offline = slot.querySelector(".cam-offline");
        const img = slot.querySelector("img[data-cam]");
        if (img) {
          img.style.display = "none";
          img.src = "";
        }
        if (offline) {
          offline.textContent = msg;
          offline.style.display = "";
        }
      }
      function setCamerasForPage(page) {
        const want = page === "grading";
        if (want === camerasActive) return;
        const container = document.getElementById("gradeCams");
        if (!want) {
          container.querySelectorAll("img[data-cam]").forEach((img) => {
            if (img._ws) {
              try { img._ws.close(); } catch(e) {}
              img._ws = null;
            }
            if (img._blobUrl) {
              URL.revokeObjectURL(img._blobUrl);
              img._blobUrl = null;
            }
            clearTimeout(img._wsRetryTimer);
            clearTimeout(img._wsTimeout);
            img.src = "";
            img.remove();
          });
          camerasActive = false;
          container.querySelectorAll(".cam-offline").forEach((el) => {
            el.style.display = "";
          });
          return;
        }
        camerasActive = true;
        startCameraStreams();
      }
      function startCameraStreams() {
        const container = document.getElementById("gradeCams");
        const cams = ["side", "top", "front"];
        const slots = container.querySelectorAll(".cam-slot");
        slots.forEach((slot, i) => {
          const cam = cams[i];
          const offline = slot.querySelector(".cam-offline");

          // Clean up existing connections
          let img = slot.querySelector("img[data-cam]");
          if (img && img._ws) {
            try { img._ws.close(); } catch(e) {}
            img._ws = null;
          }
          if (img && img._blobUrl) {
            URL.revokeObjectURL(img._blobUrl);
            img._blobUrl = null;
          }

          if (!img) {
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
            };
            slot.insertBefore(img, slot.firstChild);
          }
          img.style.display = "";

          if (cameraColdBlocked) {
            setCameraBlockedState(slot, getCameraBlockedMessage());
            img.removeAttribute("src");
            return;
          }

          // Try WebSocket first, fall back to MJPEG
          _connectCameraWS(img, cam, offline);
        });
      }

      function _connectCameraWS(img, cam, offlineEl) {
        // WS disabled — go straight to MJPEG
        _fallbackToMJPEG(img, cam, offlineEl);
      }

      function _fallbackToMJPEG(img, cam, offlineEl) {
        // Clean up WS state
        if (img._ws) {
          try { img._ws.close(); } catch(e) {}
          img._ws = null;
        }
        clearTimeout(img._wsTimeout);
        clearTimeout(img._wsRetryTimer);

        if (!piConnected) {
          img.removeAttribute("src");
          if (offlineEl) {
            offlineEl.textContent = "Waiting for grader...";
            offlineEl.style.display = "";
          }
          return;
        }

        // Standard MJPEG stream
        HerdAuth.setPiImageSource(img, {
          kind: "stream",
          view: cam,
        }).catch(() => {
          if (typeof img.onerror === "function") img.onerror.call(img);
        });

        img.onerror = function () {
          this.style.display = "none";
          if (offlineEl) {
            offlineEl.textContent = cameraColdBlocked
              ? getCameraBlockedMessage()
              : piConnected
              ? "Stream interrupted — retrying..."
              : "Waiting for grader...";
            offlineEl.style.display = "";
          }
          if (cameraColdBlocked) return;
          clearTimeout(this._retryTimer);
          this._retryTimer = setTimeout(() => {
            if (currentPage === "grading" && piConnected) {
              _connectCameraWS(this, cam, offlineEl);
            }
          }, 10000);
        };
      }
      function refreshCameraTokens() {
        if (!camerasActive || currentPage !== "grading") return;
      }
      setInterval(refreshCameraTokens, 10 * 60 * 1000);

      // PI CONNECTION CHECK
      async function checkPiConnection() {
        const el = document.getElementById("piStatus");
        try {
          const controller = new AbortController();
          const timer = setTimeout(() => controller.abort(), 5000);
          const resp = await HerdAuth.fetch("/api/prod/status", {
            signal: controller.signal,
          });
          clearTimeout(timer);
          const data = await resp.json().catch(() => ({}));
          const wasOffline = !piConnected;
          const wasColdBlocked = cameraColdBlocked;
          piConnected = true;
          cameraSafety = data.camera_safety || null;
          cameraColdBlocked = !!cameraSafety?.cold_gate_active;
          cameraColdMessage = cameraColdBlocked
            ? cameraSafety?.message || CAMERA_COLD_MESSAGE
            : CAMERA_COLD_MESSAGE;
          el.innerHTML =
            '<span class="dot green"></span><span>Grader connected</span>';
          if (wasOffline) {
            updateGradeButtonState();
            if (currentPage === "grading" && camerasActive)
              startCameraStreams();
            else if (currentPage === "grading") setCamerasForPage("grading");
          } else if (currentPage === "grading" && wasColdBlocked !== cameraColdBlocked) {
            startCameraStreams();
          }
        } catch (e) {
          const wasOnline = piConnected;
          piConnected = false;
          cameraSafety = null;
          cameraColdBlocked = false;
          cameraColdMessage = CAMERA_COLD_MESSAGE;
          el.innerHTML =
            '<span class="dot red"></span><span>Grader offline</span>';
          if (wasOnline) updateGradeButtonState();
        }
      }
      function updateGradeButtonState() {
        const btn = document.getElementById("gradeBtn");
        if (currentPage !== "grading") return;
        btn.disabled = !piConnected;
        btn.title = piConnected ? "" : "Grader is offline — cannot grade";
      }
      setInterval(checkPiConnection, 10000);

      // NAV
