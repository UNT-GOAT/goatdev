      // VIEW FOCUS: STREAMING & FOCUS CONTROL
      // ============================================================

      function buildFocusGrid() {
        const container = document.getElementById("focusCams");
        container.innerHTML = "";

        CAMERAS.forEach((cam) => {
          const card = document.createElement("div");
          card.className = "panel p-14";
          card.innerHTML =
            '<div class="flex justify-between items-center mb-10">' +
            '<span style="font-size:14px;font-weight:600;text-transform:capitalize">' +
            cam +
            "</span>" +
            '<span class="heater-status off" id="focus-status-' +
            cam +
            '">Offline</span>' +
            "</div>" +
            '<div class="cam-slot" id="focus-cam-' +
            cam +
            '" style="cursor:pointer" onclick="openLightbox(this)">' +
            '<span class="cam-label">' +
            cam +
            "</span>" +
            '<div class="cam-offline" id="focus-offline-' +
            cam +
            '">Waiting for grader...</div>' +
            "</div>" +
            '<div style="margin-top:10px">' +
            '<div class="flex items-center gap-8 mb-10">' +
            '<span class="field-label" style="margin:0;flex-shrink:0">Focus</span>' +
            '<input type="range" min="0" max="1023" value="' +
            (focusValues[cam] || 200) +
            '" ' +
            'id="focus-slider-' +
            cam +
            '" ' +
            "oninput=\"onFocusSliderInput('" +
            cam +
            "', this.value)\" " +
            "onchange=\"setFocus('" +
            cam +
            "', this.value)\" />" +
            '<span class="text-mono" style="font-size:12px;color:var(--accent);min-width:32px;text-align:right" ' +
            'id="fval-' +
            cam +
            '">' +
            (focusValues[cam] || 200) +
            "</span>" +
            '<button class="btn btn-sm" style="padding:4px 8px;font-size:10px" ' +
            'id="af-btn-' +
            cam +
            '" onclick="autoFocus(\'' +
            cam +
            "')\">AF</button>" +
            "</div>" +
            '<div id="fmsg-' +
            cam +
            '" style="font-size:10px;color:var(--text-faint);min-height:14px"></div>' +
            "</div>";
          container.appendChild(card);
        });
      }

      function onFocusSliderInput(cam, val) {
        document.getElementById("fval-" + cam).textContent = val;
        focusValues[cam] = parseInt(val);

        // Disable AF if it was active
        if (_afActive[cam]) {
          _afActive[cam] = false;
          const btn = document.getElementById("af-btn-" + cam);
          const msgEl = document.getElementById("fmsg-" + cam);
          btn.textContent = "AF";
          btn.style.color = "";
          msgEl.textContent = "";
        }

        if (_focusThrottleTimers[cam]) return;
        _focusThrottleTimers[cam] = setTimeout(() => {
          _focusThrottleTimers[cam] = null;
          HerdAuth.fetch(
            "/api/viewfocus/focus/" + cam + "/" + focusValues[cam],
          ).catch(() => {});
        }, 100);
      }

      // --- MJPEG Streaming (mirrors dashboard setCamerasForPage / startCameraStreams) ---

      function startFocusStreams() {
        if (focusStreamsActive) return;
        focusStreamsActive = true;

        CAMERAS.forEach((cam) => {
          const slot = document.getElementById("focus-cam-" + cam);
          if (!slot) return;
          const offline = document.getElementById("focus-offline-" + cam);

          // Clean up existing
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
            updateCamStatus(cam, "Streaming");
          };
          img.onerror = function () {
            this.style.display = "none";
            if (offline) {
              offline.textContent = piConnected
                ? "Stream interrupted — retrying..."
                : "Waiting for grader...";
              offline.style.display = "";
            }
            updateCamStatus(cam, "Offline", true);
            clearTimeout(this._retryTimer);
            this._retryTimer = setTimeout(() => {
              if (
                currentPage === "viewfocus" &&
                piConnected &&
                focusStreamsActive
              ) {
                _connectFocusStream(this, cam, offline);
              }
            }, 10000);
          };
          slot.insertBefore(img, slot.firstChild);
          _connectFocusStream(img, cam, offline);
        });
      }

      function _connectFocusStream(img, cam, offlineEl) {
        if (!piConnected) {
          img.removeAttribute("src");
          img.style.display = "none";
          if (offlineEl) {
            offlineEl.textContent = "Waiting for grader...";
            offlineEl.style.display = "";
          }
          updateCamStatus(cam, "Offline", true);
          return;
        }
        img.style.display = "";
        HerdAuth.setPiImageSource(img, {
          kind: "stream",
          view: cam,
        }).catch(() => {
          if (typeof img.onerror === "function") img.onerror.call(img);
        });
      }

      function stopFocusStreams() {
        if (!focusStreamsActive) return;
        focusStreamsActive = false;

        CAMERAS.forEach((cam) => {
          const slot = document.getElementById("focus-cam-" + cam);
          if (!slot) return;
          const img = slot.querySelector("img[data-cam]");
          if (img) {
            clearTimeout(img._retryTimer);
            img.src = "";
            img.remove();
          }
          const offline = document.getElementById("focus-offline-" + cam);
          if (offline) {
            offline.textContent = "Waiting for grader...";
            offline.style.display = "";
          }
          updateCamStatus(cam, "Offline", true);
        });
      }

      function updateCamStatus(cam, text, isErr) {
        const el = document.getElementById("focus-status-" + cam);
        if (!el) return;
        el.className = isErr ? "heater-status failsafe" : "heater-status on";
        el.textContent = text;
      }

      // Refresh MJPEG tokens periodically (mirrors dashboard refreshCameraTokens)
      function refreshFocusTokens() {
        if (!focusStreamsActive || currentPage !== "viewfocus") return;
      }
      setInterval(refreshFocusTokens, 10 * 60 * 1000);

      // ============================================================
      // FOCUS CONTROL API
      // ============================================================

      async function setFocus(cam, val) {
        focusValues[cam] = parseInt(val);
        document.getElementById("fval-" + cam).textContent = val;
        // Clear any pending throttle and send the final value
        clearTimeout(_focusThrottleTimers[cam]);
        _focusThrottleTimers[cam] = null;
        try {
          await HerdAuth.fetch("/api/viewfocus/focus/" + cam + "/" + val);
        } catch (e) {}
      }

      async function autoFocus(cam) {
        const btn = document.getElementById("af-btn-" + cam);
        const msgEl = document.getElementById("fmsg-" + cam);

        try {
          const r = await HerdAuth.fetch("/api/viewfocus/autofocus/" + cam, {
            method: "POST",
          });
          if (r.ok) {
            _afActive[cam] = true;
            btn.textContent = "AF ●";
            btn.style.color = "var(--accent)";
            msgEl.textContent = "AF active — touch slider to lock";
            msgEl.style.color = "var(--accent)";
          }
        } catch (e) {
          msgEl.textContent = "AF failed";
          msgEl.style.color = "var(--red)";
          setTimeout(() => {
            msgEl.textContent = "";
          }, 3000);
        }
      }

      async function loadFocusFromStatus() {
        try {
          const r = await HerdAuth.fetch("/api/viewfocus/status");
          if (r.ok) {
            const data = await r.json();
            if (data.settings) {
              CAMERAS.forEach((cam) => {
                if (data.settings[cam] !== undefined) {
                  focusValues[cam] = data.settings[cam];
                  const slider = document.getElementById("focus-slider-" + cam);
                  const valEl = document.getElementById("fval-" + cam);
                  if (slider) slider.value = focusValues[cam];
                  if (valEl) valEl.textContent = focusValues[cam];
                }
              });
            }
          }
        } catch (e) {}
      }

      async function captureTestShots() {
        const status = document.getElementById("focusActionStatus");
        status.textContent = "Capturing test shots...";
        status.style.color = "var(--purple)";

        let ok = 0;
        for (const cam of CAMERAS) {
          try {
            const r = await HerdAuth.fetch(
              "/api/viewfocus/capture/burst/" + cam,
              {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ count: 1, interval_ms: 100 }),
              },
            );
            if (r.ok) ok++;
          } catch (e) {}
        }
        status.textContent = ok + "/" + CAMERAS.length + " test shots captured";
        status.style.color =
          ok === CAMERAS.length ? "var(--accent)" : "var(--orange)";
        if (ok > 0) showToast("success", ok + " test shot(s) captured");
        else
          showToast("error", "No test shots captured — grader may be offline");
        setTimeout(() => {
          status.textContent = "";
        }, 4000);
      }

      // ============================================================
