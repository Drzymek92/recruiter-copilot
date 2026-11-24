// Second Chair workspace pages — shared helpers (D14: hand-written, no framework, no build step).
// Loaded in <head> (not deferred) so the display prefs apply before first paint.
//
// Display prefs: the cockpit's ONE key (D14 amendment 11-20): font / theme / text size only, never
// job, candidate or session data. Every storage access is guarded; blocked storage = defaults.
"use strict";
window.scPrefs = (function () {
  const KEY = "recruiter-copilot.cockpit.prefs";
  const CHOICES = {
    font: ["sans", "serif", "mono"],
    theme: ["system", "light", "dark", "contrast"],
    size: ["s", "m", "l", "xl"],
  };
  const DEFAULTS = { font: "sans", theme: "system", size: "m" };
  let raw = {};
  let storageOk = true;
  function clean(src) {
    const p = Object.assign({}, DEFAULTS);
    if (src && typeof src === "object") {
      Object.keys(CHOICES).forEach((k) => { if (CHOICES[k].includes(src[k])) p[k] = src[k]; });
    }
    return p;
  }
  function load() {
    try {
      raw = JSON.parse(window.localStorage.getItem(KEY) || "{}") || {};
      if (typeof raw !== "object") raw = {};
    } catch (e) {
      storageOk = false;
      raw = {};
    }
    return clean(raw);
  }
  function apply(p) {
    const root = document.documentElement;
    root.setAttribute("data-font", p.font);
    root.setAttribute("data-theme", p.theme);
    root.setAttribute("data-size", p.size);
  }
  let state = load();
  apply(state);
  return {
    CHOICES,
    get: () => Object.assign({}, state),
    set(patch) {
      state = clean(Object.assign({}, state, patch));
      apply(state);
      // keep the cockpit's own fields (its column split) — same key, display values only
      const merged = Object.assign({}, raw, state);
      try {
        window.localStorage.setItem(KEY, JSON.stringify(merged));
        raw = merged;
        storageOk = true;
      } catch (e) {
        storageOk = false;
      }
      return state;
    },
    storageOk: () => storageOk,
  };
})();

function escapeHtml(s) {
  return String(s == null ? "" : s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// fetch JSON; a non-2xx reply rejects with {status, message, errors}
async function api(method, url, body) {
  const opts = { method: method, headers: {} };
  if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  let resp;
  try {
    resp = await fetch(url, opts);
  } catch (e) {
    throw { status: 0, message: "the workspace app is not reachable — is it still running?", errors: [] };
  }
  let data = null;
  try { data = await resp.json(); } catch (e) { data = null; }
  if (!resp.ok) {
    throw {
      status: resp.status,
      message: (data && data.error) || ("request failed (" + resp.status + ")"),
      errors: (data && Array.isArray(data.errors)) ? data.errors : [],
    };
  }
  return data;
}

// one file as a raw request body (no multipart parser on the server)
async function uploadFile(url, file, maxMb) {
  if (maxMb && file.size > maxMb * 1048576) {
    throw { status: 413, message: file.name + " is larger than " + maxMb + " MB", errors: [] };
  }
  let resp;
  try {
    resp = await fetch(url, {
      method: "PUT",
      body: file,
      headers: { "X-Filename": encodeURIComponent(file.name), "Content-Type": "application/octet-stream" },
    });
  } catch (e) {
    throw { status: 0, message: "the workspace app is not reachable", errors: [] };
  }
  let data = null;
  try { data = await resp.json(); } catch (e) { data = null; }
  if (!resp.ok) throw { status: resp.status, message: (data && data.error) || "upload failed", errors: [] };
  return data;
}

function setStatus(el, kind, text) {
  if (!el) return;
  el.className = "status" + (kind ? " " + kind : "");
  el.textContent = text || "";
}

function errText(e) {
  return (e && e.message) ? e.message : String(e);
}

// A native modal <dialog>: focus is trapped, Esc cancels, focus returns to the opener.
// Resolves true only on the explicit confirm button.
function confirmDialog(title, bodyText, confirmLabel, pathText) {
  return new Promise((resolve) => {
    const opener = document.activeElement;
    const dlg = document.createElement("dialog");
    dlg.setAttribute("aria-labelledby", "cd-title");
    dlg.innerHTML =
      '<h2 id="cd-title">' + escapeHtml(title) + '</h2>' +
      '<p>' + escapeHtml(bodyText) + '</p>' +
      (pathText ? '<p class="path">' + escapeHtml(pathText) + '</p>' : '') +
      '<div class="actions"><button type="button" class="danger" data-act="yes">' +
      escapeHtml(confirmLabel || "Delete") + '</button>' +
      '<button type="button" data-act="no" autofocus>Cancel</button></div>';
    document.body.appendChild(dlg);
    let answered = false;
    function done(v) {
      if (answered) return;
      answered = true;
      if (dlg.open) dlg.close();
      dlg.remove();
      if (opener && typeof opener.focus === "function") opener.focus();
      resolve(v);
    }
    dlg.addEventListener("click", (ev) => {
      const b = ev.target.closest("button[data-act]");
      if (b) done(b.dataset.act === "yes");
    });
    dlg.addEventListener("cancel", () => done(false));
    dlg.addEventListener("close", () => done(false));
    dlg.showModal();
    const no = dlg.querySelector('[data-act="no"]');
    if (no) no.focus();
  });
}

// Header "Display" control (theme / text size / font), the cockpit's key.
function mountDisplayControl(host) {
  if (!host) return;
  const p = window.scPrefs.get();
  const optsHtml = (name, values, labels) => values.map((v, i) =>
    '<option value="' + escapeHtml(v) + '"' + (p[name] === v ? " selected" : "") + '>' +
    escapeHtml(labels[i]) + '</option>').join("");
  host.innerHTML =
    '<details class="display"><summary>Display</summary><div class="seg" style="margin-top:6px">' +
    '<label class="inline">Theme <select data-pref="theme">' +
    optsHtml("theme", window.scPrefs.CHOICES.theme, ["System", "Light", "Dark", "High contrast"]) +
    '</select></label>' +
    '<label class="inline">Text <select data-pref="size">' +
    optsHtml("size", window.scPrefs.CHOICES.size, ["Small", "Medium", "Large", "Extra large"]) +
    '</select></label>' +
    '<label class="inline">Font <select data-pref="font">' +
    optsHtml("font", window.scPrefs.CHOICES.font, ["Sans", "Serif", "Mono"]) +
    '</select></label></div></details>';
  host.addEventListener("change", (ev) => {
    const sel = ev.target.closest("select[data-pref]");
    if (!sel) return;
    const patch = {};
    patch[sel.dataset.pref] = sel.value;
    window.scPrefs.set(patch);
  });
}

function langLabel(code) {
  const names = { pl: "Polish", en: "English", de: "German", uk: "Ukrainian" };
  return names[code] ? names[code] + " (" + code + ")" : String(code || "");
}

function todayIso() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  return d.getFullYear() + "-" + pad(d.getMonth() + 1) + "-" + pad(d.getDate());
}
