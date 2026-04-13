      function showApp(user) {
        allAnimals = [];
        allProviders = [];
        allGrades = [];
        providerMap = {};
        loadErrors = [];
        selectedAnimalId = null;
        selectedAnimalType = null;
        pendingGradeResult = null;
        _gradeAnimalCreated = false;
        _gradeExistingId = null;
        _gradeSessionId = null;
        _gradeAnalysisKey = null;
        _pendingNewAnimalSession = null;
        _reviewGradeId = null;
        _currentUser = user;
        document.getElementById("app").style.display = "block";
        document.getElementById("topbarUser").textContent =
          user.username + " (" + user.role + ")";
        document.getElementById("footerUser").textContent = user.username;
        document.getElementById("profileUsername").textContent = user.username;
        document.getElementById("profileRole").textContent = user.role;
        if (user.role === "admin") {
          document.getElementById("adminSection").style.display = "";
          loadUsers();
        } else {
          document.getElementById("adminSection").style.display = "none";
        }
        _showDevUploadIfDev();
        loadAll();
        checkPiConnection();
      }
      async function loadAll() {
        loadErrors = [];
        try {
          await loadProviders();
        } catch (err) {
          loadErrors.push("Providers: " + err.message);
        }
        await Promise.all([
          loadGoats().catch((err) => {
            loadErrors.push("Goats: " + err.message);
          }),
          loadChickens().catch((err) => {
            loadErrors.push("Chickens: " + err.message);
          }),
          loadLambs().catch((err) => {
            loadErrors.push("Lambs: " + err.message);
          }),
          loadGrades().catch((err) => {
            loadErrors.push("Grades: " + err.message);
          }),
        ]);
        try {
          await loadPendingNewAnimalSession();
        } catch (err) {
          loadErrors.push("Pending grade session: " + err.message);
        }
        updateDashboard();
        filterAnimals();
        filterProviders();
        renderGradeHistory();
        computeNextGradeId();
        if (typeof restorePendingNewAnimalReview === "function") {
          restorePendingNewAnimalReview();
        }
        if (loadErrors.length === 0) setDbStatus("ok");
        else if (allAnimals.length > 0 || allProviders.length > 0) {
          setDbStatus("partial");
          showLoadErrors();
        } else {
          setDbStatus("error");
          showLoadErrors();
        }
      }
      function setDbStatus(state) {
        const el = document.getElementById("dbStatus");
        if (state === "ok")
          el.innerHTML =
            '<span class="dot green"></span><span>User Connected</span>';
        else if (state === "partial")
          el.innerHTML = '<span class="dot orange"></span><span>Partial</span>';
        else
          el.innerHTML = '<span class="dot red"></span><span>DB Error</span>';
      }
      function showLoadErrors() {
        const el = document.getElementById("loadErrors");
        el.style.display = "";
        el.innerHTML = loadErrors
          .map((e) => '<div class="load-error">' + esc(e) + "</div>")
          .join("");
      }

      // DATA LOADING
      async function dbFetch(path, opts = {}) {
        const r = await HerdAuth.fetch("/db" + path, opts);
        if (!r.ok) throw new Error(r.status + " on " + path);
        return r.json();
      }
      async function authFetch(path, opts) {
        return HerdAuth.fetch(path, opts);
      }
      async function loadProviders() {
        allProviders = await dbFetch("/providers");
        providerMap = {};
        allProviders.forEach((p) => {
          providerMap[p.id] = p;
        });
      }
      async function loadGoats() {
        const rows = await dbFetch("/goats");
        allAnimals = allAnimals
          .filter((a) => a.type !== "goat")
          .concat(
            rows.map((r) => ({
              ...r,
              type: "goat",
              description: r.description || r.hook_id || "",
              providerName: providerMap[r.prov_id]?.name || "—",
            })),
          );
      }
      async function loadChickens() {
        const rows = await dbFetch("/chickens");
        allAnimals = allAnimals
          .filter((a) => a.type !== "chicken")
          .concat(
            rows.map((r) => ({
              ...r,
              type: "chicken",
              grade: null,
              description: "",
              providerName: providerMap[r.prov_id]?.name || "—",
            })),
          );
      }
      async function loadLambs() {
        const rows = await dbFetch("/lambs");
        allAnimals = allAnimals
          .filter((a) => a.type !== "lamb")
          .concat(
            rows.map((r) => ({
              ...r,
              type: "lamb",
              description: r.description || "",
              providerName: providerMap[r.prov_id]?.name || "—",
            })),
          );
      }
      async function loadGrades() {
        allGrades = await dbFetch("/grading?limit=100");
      }

      // CAMERAS

      HerdAuth.requireAuth().then((user) => showApp(user));
