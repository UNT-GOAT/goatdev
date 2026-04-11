      const TITLES = {
        dashboard: "Dashboard",
        animals: "Animals",
        providers: "Providers",
        grading: "Grading",
        logs: "Logs",
        profile: "Profile",
      };
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
          document.querySelector(".content").style.overflow =
             window.innerWidth <= 640 ? "auto" : (pg === "grading" || pg === "profile" ? "auto" : "hidden");
          setCamerasForPage(pg);
          if (pg === "grading") computeNextGradeId();
          if (pg === "logs") loadAuditLogs();
          if (pg === "profile" && _currentUser?.role === "admin") loadUsers();
        });
      });
      function toggleSidebar() {
        if (window.innerWidth <= 640) return;
        document.getElementById("sidebar").classList.toggle("collapsed");
      }

      window.addEventListener("resize", () => {
        document.querySelector(".content").style.overflow =
          window.innerWidth <= 640 ? "auto" : (currentPage === "grading" || currentPage === "profile" ? "auto" : "hidden");
      });

      // DASHBOARD
      function updateDashboard() {
        const goats = allAnimals.filter((a) => a.type === "goat"),
          chickens = allAnimals.filter((a) => a.type === "chicken"),
          lambs = allAnimals.filter((a) => a.type === "lamb");
        document.getElementById("statGoats").textContent = goats.length;
        document.getElementById("statGoatsSub").textContent =
          goats.filter((g) => g.grade).length + " graded";
        document.getElementById("statChickens").textContent = chickens.length;
        document.getElementById("statChickensSub").textContent = "in database";
        document.getElementById("statLambs").textContent = lambs.length;
        document.getElementById("statLambsSub").textContent =
          lambs.filter((l) => l.grade).length + " graded";
        document.getElementById("statProviders").textContent =
          allProviders.length;
        const gradeEl = document.getElementById("dashRecentGrades");
        if (!allGrades.length) {
          gradeEl.innerHTML =
            '<div style="padding:8px;color:var(--text-muted);font-size:12px">No grades yet</div>';
        } else {
          gradeEl.innerHTML = allGrades
            .map(
              (g) =>
                '<div class="row-item flex items-center justify-between"><div><span style="color:var(--text-primary);font-weight:600;font-size:13px">' +
                g.serial_id +
                "</span>" +
                (g.species
                  ? '<span style="margin-left:6px">' +
                    typeBadge(g.species) +
                    "</span>"
                  : "") +
                (g.description
                  ? '<span style="margin-left:4px;color:var(--text-muted);font-size:10px">' +
                    esc(g.description) +
                    "</span>"
                  : "") +
                '</div><div class="flex items-center gap-8">' +
                gradeBadge(g.grade) +
                '<span style="color:var(--text-faint);font-size:10px">' +
                fmtDate(g.graded_at) +
                "</span></div></div>",
            )
            .join("");
        }
        const provEl = document.getElementById("dashProviders");
        if (!allProviders.length) {
          provEl.innerHTML =
            '<div style="padding:8px;color:var(--text-muted);font-size:12px">No providers</div>';
        } else {
          provEl.innerHTML = allProviders
            .map((p) => {
              const cnt = allAnimals.filter((a) => a.prov_id === p.id).length;
              return (
                '<div class="row-item flex items-center justify-between"><span style="color:var(--text-primary);font-weight:600;font-size:13px">' +
                esc(p.name) +
                '</span><span class="badge badge-prime" style="font-size:10px">' +
                cnt +
                " head</span></div>"
              );
            })
            .join("");
        }
      }


      // ANIMALS
