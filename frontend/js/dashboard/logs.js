      // AUDIT LOGS
      // ============================================================
      async function loadAuditLogs() {
        const body = document.getElementById("logTableBody");
        body.innerHTML =
          '<div class="loading-row"><span class="spinner"></span></div>';
        const user = document.getElementById("logUserFilter").value;
        const action = document.getElementById("logActionFilter").value;
        const resource = document.getElementById("logResourceFilter").value;
        let qs = "limit=500";
        if (user) qs += "&username=" + encodeURIComponent(user);
        if (action) qs += "&action=" + encodeURIComponent(action);
        if (resource) qs += "&resource_type=" + encodeURIComponent(resource);
        try {
          const r = await dbFetch("/audit-logs?" + qs);
          _auditLogs = r.logs || [];
          // Populate user filter if empty
          const uf = document.getElementById("logUserFilter");
          if (uf.options.length <= 1) {
            const users = [
              ...new Set(_auditLogs.map((l) => l.username)),
            ].sort();
            users.forEach((u) => {
              const o = document.createElement("option");
              o.value = u;
              o.textContent = u;
              uf.appendChild(o);
            });
          }
          renderAuditLogs();
        } catch (err) {
          body.innerHTML = '<div class="no-results">Failed to load logs</div>';
        }
      }
      function filterLogs() {
        renderAuditLogs();
      }
      function clearLogSearch() {
        document.getElementById("logSearch").value = "";
        document.getElementById("logClear").classList.remove("visible");
        renderAuditLogs();
        document.getElementById("logSearch").focus();
      }
      function renderAuditLogs() {
        const q = (document.getElementById("logSearch")?.value || "")
          .trim()
          .toLowerCase();
        document
          .getElementById("logClear")
          .classList.toggle("visible", q.length > 0);
        let logs = _auditLogs;
        if (q) {
          logs = logs.filter((l) =>
            [
              l.username,
              l.action,
              l.resource_type,
              l.resource_id || "",
              JSON.stringify(l.detail || ""),
            ]
              .join(" ")
              .toLowerCase()
              .includes(q),
          );
        }
        document.getElementById("logCount").textContent =
          logs.length + " entries";
        const body = document.getElementById("logTableBody");
        if (!logs.length) {
          body.innerHTML = '<div class="no-results">No log entries found</div>';
          return;
        }
        const actionColors = {
          create: "var(--accent)",
          update: "var(--blue)",
          delete: "var(--red)",
        };
        body.innerHTML = logs
          .map((l) => {
            const ts = l.timestamp
              ? new Date(l.timestamp).toLocaleString("en-US", {
                  month: "short",
                  day: "numeric",
                  hour: "numeric",
                  minute: "2-digit",
                  hour12: true,
                })
              : "—";
            const actionColor = actionColors[l.action] || "var(--text-muted)";
            const desc = _formatAuditDetail(l);
            return (
              '<div style="padding:10px 16px;border-bottom:1px solid var(--border-subtle);display:flex;gap:12px;align-items:flex-start">' +
              '<div style="flex:1;min-width:0">' +
              '<div style="display:flex;align-items:center;gap:8px;margin-bottom:2px">' +
              '<span style="font-size:12px;font-weight:600;color:var(--text-primary)">' +
              esc(l.username) +
              "</span>" +
              '<span style="font-size:10px;font-weight:600;color:' +
              actionColor +
              ';text-transform:uppercase;letter-spacing:0.5px">' +
              esc(l.action) +
              "</span>" +
              '<span style="font-size:10px;color:var(--text-muted)">' +
              esc(l.resource_type) +
              (l.resource_id ? " #" + esc(l.resource_id) : "") +
              "</span>" +
              "</div>" +
              (desc
                ? '<div style="font-size:11px;color:var(--text-muted);line-height:1.6">' +
                  desc +
                  "</div>"
                : "") +
              "</div>" +
              '<div style="flex-shrink:0;font-size:10px;color:var(--text-faint);white-space:nowrap">' +
              ts +
              "</div>" +
              "</div>"
            );
          })
          .join("");
      }
      function _formatAuditDetail(log) {
        if (!log.detail) return "";
        const d = log.detail;
        const lines = [];
        function fmtVal(v) {
          if (v === null || v === undefined)
            return '<span style="color:var(--text-faint);font-style:italic">none</span>';
          if (typeof v === "boolean") return v ? "yes" : "no";
          if (typeof v === "object") return JSON.stringify(v);
          return esc(String(v));
        }
        function fmtKey(k) {
          return k.replace(/_/g, " ").replace(/\bid\b/g, "ID");
        }
        function provLookup(id) {
          if (!id) return null;
          const p = providerMap[id];
          return p ? p.name + " (#" + id + ")" : "#" + id;
        }
        if (log.action === "create") {
          const rec = d.record || d;
          if (rec.serial_id)
            lines.push("Serial <strong>#" + rec.serial_id + "</strong>");
          if (rec.name)
            lines.push("Name: <strong>" + esc(rec.name) + "</strong>");
          if (rec.species)
            lines.push(
              esc(rec.species.charAt(0).toUpperCase() + rec.species.slice(1)),
            );
          if (rec.description) lines.push(esc(rec.description));
          if (rec.grade)
            lines.push("Grade: <strong>" + esc(rec.grade) + "</strong>");
          if (rec.live_weight)
            lines.push("Live wt: " + rec.live_weight + " lbs");
          if (rec.hang_weight)
            lines.push("Hang wt: " + rec.hang_weight + " lbs");
          if (rec.prov_id) {
            const pn = provLookup(rec.prov_id);
            if (pn) lines.push("Provider: " + esc(pn));
          }
          if (rec.kill_date) lines.push("Kill: " + rec.kill_date);
          if (rec.phone) lines.push(esc(rec.phone));
          if (rec.email) lines.push(esc(rec.email));
          if (rec.company) lines.push(esc(rec.company));
        } else if (log.action === "update") {
          const changed = d.changed || d;
          const after = d.after || null;
          if (after && after.name && log.resource_type === "provider")
            lines.push("<strong>" + esc(after.name) + "</strong>");
          else if (after && after.serial_id)
            lines.push("Serial #" + after.serial_id);
          const keys = Object.keys(changed).filter(
            (k) => k !== "id" && k !== "serial_id" && !k.endsWith("_at"),
          );
          keys.forEach((k) => {
            const newVal = changed[k];
            const label = fmtKey(k);
            if (k === "prov_id") {
              const pn = provLookup(newVal);
              lines.push(
                esc(label) +
                  " → <strong>" +
                  esc(pn || String(newVal)) +
                  "</strong>",
              );
            } else if (k === "password") {
              lines.push("password → <strong>••••••</strong>");
            } else {
              lines.push(
                esc(label) + " → <strong>" + fmtVal(newVal) + "</strong>",
              );
            }
          });
        } else if (log.action === "delete") {
          if (d.serial_id)
            lines.push("Serial <strong>#" + d.serial_id + "</strong>");
          if (d.deleted) lines.push("ID: " + d.deleted);
          if (d.name) lines.push(esc(d.name));
        }
        return lines.join(" · ");
      }

      // TOAST
