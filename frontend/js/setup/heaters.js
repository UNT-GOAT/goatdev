      // ============================================================
      function startHeaterPolling() {
        if (_heaterInterval) return;
        fetchHeaterStatus();
        fetchHeaterHistory();
        _heaterInterval = setInterval(fetchHeaterStatus, 2000);
        _heaterHistoryInterval = setInterval(fetchHeaterHistory, 10000);
      }

      function stopHeaterPolling() {
        if (_heaterInterval) {
          clearInterval(_heaterInterval);
          _heaterInterval = null;
        }
        if (_heaterHistoryInterval) {
          clearInterval(_heaterHistoryInterval);
          _heaterHistoryInterval = null;
        }
      }

      async function fetchHeaterStatus() {
        if (!piConnected) {
          // Show offline state in the heater grid
          const grid = document.getElementById("heaterGrid");
          if (grid.children.length === 0) {
            grid.innerHTML =
              '<div style="grid-column:1/-1;padding:30px;text-align:center;color:var(--text-muted);font-size:13px">' +
              '<div class="dot red" style="display:inline-block;margin-right:6px"></div>' +
              "Pi is offline — heater data unavailable. Waiting for connection..." +
              "</div>";
          }
          return;
        }
        try {
          const r = await HerdAuth.fetch("/api/heater/status");
          if (!r.ok) throw new Error("HTTP " + r.status);
          const data = await r.json();
          if (data.thresholds) {
            document.getElementById("threshOn").textContent =
              "ON < " + data.thresholds.on_below + "°F";
            document.getElementById("threshOff").textContent =
              "OFF > " + data.thresholds.off_above + "°F";
          }
          renderHeaters(data.cameras || {});
        } catch (e) {
          // Show error in grid if it's empty
          const grid = document.getElementById("heaterGrid");
          if (
            grid.children.length === 0 ||
            grid.querySelector("[data-offline-msg]")
          ) {
            grid.innerHTML =
              '<div data-offline-msg style="grid-column:1/-1;padding:30px;text-align:center;color:var(--red);font-size:12px">' +
              "Failed to load heater data: " +
              (e.message || "unknown error") +
              "</div>";
          }
        }
      }

      function renderHeaters(cameras) {
        const grid = document.getElementById("heaterGrid");
        const names = Object.keys(cameras).sort();
        if (grid.children.length !== names.length) {
          grid.innerHTML = "";
          names.forEach((name) => {
            const card = document.createElement("div");
            card.className = "heater-card";
            card.id = "hcard-" + name;
            card.innerHTML =
              '<div class="flex justify-between items-center mb-10">' +
              '<span style="font-size:14px;font-weight:600">' +
              name.replace("camera", "Cam ") +
              "</span>" +
              '<span class="heater-status off" id="hstatus-' +
              name +
              '">OFF</span>' +
              "</div>" +
              '<div class="temp-display warm" id="htemp-' +
              name +
              '">--°F</div>' +
              '<div id="hmeta-' +
              name +
              '" style="font-size:10px;color:var(--text-faint);margin-bottom:12px;line-height:1.8"></div>' +
              '<div class="flex gap-8">' +
              '<button class="override-btn" onclick="setHeaterOverride(\'' +
              name +
              "','on')\">Force ON</button>" +
              '<button class="override-btn active-auto" onclick="setHeaterOverride(\'' +
              name +
              "','auto')\">Auto</button>" +
              '<button class="override-btn" onclick="setHeaterOverride(\'' +
              name +
              "','off')\">Force OFF</button>" +
              "</div>";
            grid.appendChild(card);
          });
        }
        names.forEach((name) => {
          const cam = cameras[name];
          const card = document.getElementById("hcard-" + name);
          const st = document.getElementById("hstatus-" + name);
          const temp = document.getElementById("htemp-" + name);
          const meta = document.getElementById("hmeta-" + name);
          card.className = "heater-card";
          if (cam.failsafe) card.classList.add("failsafe");
          else if (cam.override !== "auto") card.classList.add("override");
          else if (cam.heater_on) card.classList.add("heating");
          if (cam.failsafe) {
            st.className = "heater-status failsafe";
            st.textContent = "FAILSAFE";
          } else if (cam.override === "on") {
            st.className = "heater-status override-on";
            st.textContent = "FORCED ON";
          } else if (cam.override === "off") {
            st.className = "heater-status override-off";
            st.textContent = "FORCED OFF";
          } else if (cam.heater_on) {
            st.className = "heater-status on";
            st.textContent = "HEATING";
          } else {
            st.className = "heater-status off";
            st.textContent = "OFF";
          }
          if (cam.temp_f !== null && cam.temp_f !== undefined) {
            temp.textContent = cam.temp_f + "°F";
            temp.className =
              "temp-display" +
              (cam.temp_f < 40 ? " cold" : cam.temp_f < 70 ? " warm" : " hot");
          } else {
            temp.textContent = "--°F";
            temp.className = "temp-display err";
          }
          let m = "";
          if (cam.last_good_temp !== null && cam.last_good_temp !== undefined)
            m += "<div>Last good: " + cam.last_good_temp + "°F</div>";
          if (cam.fail_count > 0)
            m +=
              '<div>Failed reads: <span style="color:var(--red)">' +
              cam.fail_count +
              "</span></div>";
          if (cam.last_change)
            m +=
              "<div>Last change: " +
              new Date(cam.last_change).toLocaleTimeString() +
              "</div>";
          meta.innerHTML = m;
          const btns = card.querySelectorAll(".override-btn");
          btns[0].className =
            "override-btn" + (cam.override === "on" ? " active-on" : "");
          btns[1].className =
            "override-btn" + (cam.override === "auto" ? " active-auto" : "");
          btns[2].className =
            "override-btn" + (cam.override === "off" ? " active-off" : "");
        });
      }

      async function setHeaterOverride(camera, state) {
        if (!piConnected) {
          showToast("error", "Pi is offline — cannot set override");
          return;
        }
        try {
          await HerdAuth.fetch("/api/heater/override", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ camera, state }),
          });
          showToast(
            "success",
            camera.replace("camera", "Cam ") + " → " + state.toUpperCase(),
          );
          fetchHeaterStatus();
        } catch (e) {
          showToast("error", "Failed to set override");
        }
      }

      async function fetchHeaterHistory() {
        if (!piConnected) return;
        try {
          const r = await HerdAuth.fetch("/api/heater/history");
          if (!r.ok) throw new Error("HTTP " + r.status);
          const data = await r.json();
          const list = document.getElementById("heaterHistory");
          const history = data.history || [];
          if (!history.length) {
            list.innerHTML =
              '<div style="color:var(--text-faint);padding:10px 0">No events yet</div>';
            return;
          }
          list.innerHTML = history
            .slice()
            .reverse()
            .map((e) => {
              const time = new Date(e.time).toLocaleTimeString();
              let cls = "";
              const evt = e.event || "";
              if (evt.includes("heater_on") || evt.includes("override_on"))
                cls = "on";
              else if (
                evt.includes("heater_off") ||
                evt.includes("override_off")
              )
                cls = "off";
              else if (evt.includes("failsafe")) cls = "failsafe";
              else if (evt.includes("override")) cls = "override";
              const extra = e.temp !== undefined ? " (" + e.temp + "°F)" : "";
              return (
                '<div class="history-entry"><span class="history-time">' +
                time +
                '</span><span class="history-cam">' +
                (e.camera || "") +
                '</span><span class="history-event ' +
                cls +
                '">' +
                evt +
                extra +
                "</span></div>"
              );
            })
            .join("");
        } catch (e) {
          const list = document.getElementById("heaterHistory");
          if (list.children.length <= 1) {
            list.innerHTML =
              '<div style="color:var(--text-faint);padding:10px 0">Could not load event history</div>';
          }
        }
      }

      // ============================================================
