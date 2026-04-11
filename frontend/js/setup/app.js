      // ============================================================
      // GLOBALS & CONFIG
      // ============================================================
      const CAMERAS = ["side", "top", "front"];
      const TITLES = {
        viewfocus: "View Focus",
        heaters: "Heaters",
        capture: "Capture",
      };

      // ============================================================
      // THEME
      // ============================================================
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

      // ============================================================
      // TOAST
      // ============================================================
      function showToast(type, msg) {
        HerdUI.showToast(type, msg, "toast");
      }

      // ============================================================
      // LIGHTBOX (mirrors dashboard pattern)
      // ============================================================
      function openLightbox(slot) {
        HerdUI.openLightbox(slot, {
          fallbackName: "camera",
          extension: ".jpg",
        });
      }
      function closeLightbox(e) {
        HerdUI.closeLightbox(e);
      }
      HerdUI.bindLightboxEscape();

      // ============================================================
      // NAV
      // ============================================================
      document.querySelectorAll(".nav-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
          const pg = btn.dataset.page;
          currentPage = pg;
          document
            .querySelectorAll(".nav-btn")
            .forEach((b) => b.classList.remove("active"));
          btn.classList.add("active");
          document
            .querySelectorAll(".page")
            .forEach((p) => p.classList.remove("active"));
          document.getElementById("page-" + pg).classList.add("active");
          document.getElementById("pageTitle").textContent = TITLES[pg] || pg;

          // Manage camera streams based on active page
          if (pg === "viewfocus") {
            startFocusStreams();
            stopCaptureCams();
            stopHeaterPolling();
          } else if (pg === "capture") {
            stopFocusStreams();
            startCaptureCams();
            stopHeaterPolling();
          } else if (pg === "heaters") {
            stopFocusStreams();
            stopCaptureCams();
            startHeaterPolling();
          } else {
            stopFocusStreams();
            stopCaptureCams();
            stopHeaterPolling();
          }
        });
      });

      // ============================================================
      // PI CONNECTION (mirrors dashboard pattern)
      // ============================================================
      async function checkPi() {
        const el = document.getElementById("piStatus");
        try {
          const c = new AbortController();
          const t = setTimeout(() => c.abort(), 5000);
          await HerdAuth.fetch("/api/prod/status", {
            signal: c.signal,
          });
          clearTimeout(t);
          const wasOffline = !piConnected;
          piConnected = true;
          el.innerHTML =
            '<span class="dot green"></span><span>Pi connected</span>';
          if (wasOffline) {
            if (currentPage === "viewfocus") startFocusStreams();
            if (currentPage === "capture") startCaptureCams();
            if (currentPage === "heaters") {
              fetchHeaterStatus();
              fetchHeaterHistory();
            }
          }
        } catch (e) {
          piConnected = false;
          el.innerHTML = '<span class="dot red"></span><span>Pi offline</span>';
        }
      }
      setInterval(checkPi, 10000);

      // ============================================================
      // AUTH + INIT
      // ============================================================
      function showApp(user) {
        _currentUser = user;
        document.getElementById("app").style.display = "block";
        document.getElementById("topbarUser").textContent =
          user.username + " (" + user.role + ")";
        document.getElementById("footerUser").textContent = user.username;
        checkPi();
        buildFocusGrid();
        loadFocusFromStatus();
        fetchCaptureCount();
        // Start focus streams since viewfocus is the default page
        startFocusStreams();
      }

      // ============================================================

      // ============================================================
      window.addEventListener("load", () => {
        HerdAuth.requireAuth().then((user) => showApp(user));
      });
