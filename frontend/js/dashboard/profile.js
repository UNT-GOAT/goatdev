      // PROFILE + USER MANAGEMENT
      // ============================================================
      async function changeOwnPassword() {
        const cur = document.getElementById("profileCurPw").value;
        const newPw = document.getElementById("profileNewPw").value;
        const cfm = document.getElementById("profileConfirmPw").value;
        if (!cur) {
          showToast("error", "Enter your current password");
          return;
        }
        if (!newPw || newPw.length < 8) {
          showToast(
            "error",
            "8+ chars with uppercase, lowercase, number, and special character",
          );
          return;
        }
        if (newPw !== cfm) {
          showToast("error", "Passwords do not match");
          return;
        }
        try {
          const r = await authFetch("/auth/change-password", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              current_password: cur,
              new_password: newPw,
            }),
          });
          if (!r.ok) {
            const err = await r.json().catch(() => ({}));
            throw new Error(err.detail || "Password change failed");
          }
          document.getElementById("profileCurPw").value = "";
          document.getElementById("profileNewPw").value = "";
          document.getElementById("profileConfirmPw").value = "";
          showToast("success", "Password updated");
        } catch (err) {
          showToast("error", err.message);
        }
      }

      async function loadUsers() {
        try {
          const r = await authFetch("/auth/users");
          if (!r.ok) throw new Error("Failed to load users");
          const data = await r.json();
          const users = data.users || [];
          const list = document.getElementById("adminUserList");
          if (!users.length) {
            list.innerHTML =
              '<div style="padding:8px;color:var(--text-muted);font-size:12px">No users</div>';
            return;
          }
          list.innerHTML = users
            .map(
              (u) =>
                '<div class="user-row"><div class="user-info"><span class="user-name">' +
                esc(u.username) +
                '</span><span class="user-role">' +
                esc(u.role) +
                '</span></div><div class="user-actions"><button class="btn btn-sm" onclick="openResetPassword(' +
                u.id +
                ",'" +
                esc(u.username) +
                "')\">Reset PW</button>" +
                (u.role !== "admin"
                  ? '<button class="btn btn-danger btn-sm" onclick="adminDeleteUser(' +
                    u.id +
                    ",'" +
                    esc(u.username) +
                    "')\">Delete</button>"
                  : "") +
                "</div></div>",
            )
            .join("");
        } catch (err) {
          document.getElementById("adminUserList").innerHTML =
            '<div style="padding:8px;color:var(--text-muted);font-size:12px">Could not load users</div>';
        }
      }

      function openCreateUser() {
        openModal("modalCreateUser");
        document.getElementById("newUserName").value = "";
        document.getElementById("newUserPw").value = "";
        document.getElementById("newUserRole").value = "operator";
      }
      async function createUser() {
        const username = document.getElementById("newUserName").value.trim();
        const password = document.getElementById("newUserPw").value;
        const role = document.getElementById("newUserRole").value;
        if (!username) {
          showToast("error", "Username is required");
          return;
        }
        if (!password || password.length < 8) {
          showToast(
            "error",
            "8+ chars with uppercase, lowercase, number, and special character",
          );
          return;
        }
        try {
          const r = await authFetch("/auth/users", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ username, password, role }),
          });
          if (!r.ok) {
            const err = await r.json().catch(() => ({}));
            throw new Error(err.detail || "Create failed");
          }
          closeModal("modalCreateUser");
          showToast("success", "User " + username + " created");
          loadUsers();
        } catch (err) {
          showToast("error", err.message);
        }
      }

      function openResetPassword(userId, username) {
        document.getElementById("resetPwUser").textContent = username;
        document.getElementById("resetPwUser").dataset.userId = userId;
        document.getElementById("resetPwValue").value = "";
        openModal("modalResetPw");
      }
      async function adminResetPassword() {
        const userId = document.getElementById("resetPwUser").dataset.userId;
        const newPw = document.getElementById("resetPwValue").value;
        if (!newPw || newPw.length < 8) {
          showToast(
            "error",
            "8+ chars with uppercase, lowercase, number, and special character",
          );
          return;
        }
        try {
          const r = await authFetch("/auth/users/" + userId, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ password: newPw }),
          });
          if (!r.ok) {
            const err = await r.json().catch(() => ({}));
            throw new Error(err.detail || "Reset failed");
          }
          closeModal("modalResetPw");
          showToast(
            "success",
            "Password reset for " +
              document.getElementById("resetPwUser").textContent,
          );
        } catch (err) {
          showToast("error", err.message);
        }
      }

      // ============================================================

      // ACCOUNT DELETION
      // ============================================================
      async function deleteOwnAccount() {
        if (
          !confirm(
            "Are you sure you want to permanently delete your account? You will be logged out immediately.",
          )
        )
          return;
        const typed = prompt("Type your username to confirm deletion:");
        if (typed !== _currentUser.username) {
          showToast("error", "Username did not match — account not deleted");
          return;
        }
        try {
          const r = await authFetch("/auth/me", { method: "DELETE" });
          if (!r.ok) {
            const err = await r.json().catch(() => ({}));
            throw new Error(err.detail || "Delete failed");
          }
          showToast("success", "Account deleted — logging out...");
          setTimeout(() => HerdAuth.logout(), 1500);
        } catch (err) {
          showToast("error", err.message);
        }
      }
      async function adminDeleteUser(userId, username) {
        if (
          !confirm(
            "Permanently delete user '" +
              username +
              "'? This cannot be undone.",
          )
        )
          return;
        try {
          const r = await authFetch("/auth/users/" + userId, {
            method: "DELETE",
          });
          if (!r.ok) {
            const err = await r.json().catch(() => ({}));
            throw new Error(err.detail || "Delete failed");
          }
          showToast("success", "User " + username + " deleted");
          loadUsers();
        } catch (err) {
          showToast("error", err.message);
        }
      }

      // BOOT
