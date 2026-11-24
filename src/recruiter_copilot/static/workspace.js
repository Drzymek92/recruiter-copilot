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

// fetch JSON; a non-2xx reply rejects with {status, message, errors, messages}:
// errors = the validator's machine form (to place each reason next to its field),
// messages[i] = errors[i] as a plain sentence for the recruiter (#2043).
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
      messages: (data && Array.isArray(data.messages)) ? data.messages : [],
    };
  }
  return data;
}

// the plain sentence for reason i (falls back to the machine form)
function plainAt(e, i) {
  return (e.messages && typeof e.messages[i] === "string" && e.messages[i]) || e.errors[i];
}

// one file as a raw request body (no multipart parser on the server); extraHeaders e.g. the
// start-interview form (X-Interview) sent together with its CV
async function uploadFile(url, file, maxMb, extraHeaders) {
  if (maxMb && file.size > maxMb * 1048576) {
    throw { status: 413, message: file.name + " is larger than " + maxMb + " MB", errors: [] };
  }
  const headers = Object.assign(
    { "X-Filename": encodeURIComponent(file.name), "Content-Type": "application/octet-stream" },
    extraHeaders || {});
  let resp;
  try {
    resp = await fetch(url, { method: "PUT", body: file, headers: headers });
  } catch (e) {
    throw { status: 0, message: "the workspace app is not reachable", errors: [] };
  }
  let data = null;
  try { data = await resp.json(); } catch (e) { data = null; }
  if (!resp.ok) {
    throw { status: resp.status, message: (data && data.error) || "upload failed", errors: [],
      field: (data && data.field) || "" };
  }
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

// #2043: a calm, dismissible list of notes (draft.warnings) — for the recruiter's judgement,
// never an error, never blocking Accept/Save. Every note is escaped. Dismiss = hide on this page
// (nothing is stored). `fallbackFocus` receives focus when the last note is dismissed.
function showNotes(host, list, heading, fallbackFocus) {
  if (!host) return;
  const items = (list || []).map(String);
  const hid = host.id ? host.id + "-h" : "";
  function draw(focusAt) {
    if (!items.length) {
      host.innerHTML = "";
      if (focusAt !== undefined && fallbackFocus) fallbackFocus.focus();
      return;
    }
    host.innerHTML =
      '<div class="notices" role="group"' + (hid ? ' aria-labelledby="' + escapeHtml(hid) + '"' : "") + '>' +
      '<div class="notices-top"><p class="notices-h"' + (hid ? ' id="' + escapeHtml(hid) + '"' : "") + '>' +
      escapeHtml(heading) + '</p>' +
      (items.length > 1 ? '<button type="button" class="small" data-note="all">Dismiss all</button>' : "") +
      '</div><ul>' +
      items.map((w, i) =>
        '<li><span class="note-text">' + escapeHtml(w) + '</span>' +
        '<button type="button" class="small" data-note="' + escapeHtml(i) + '" aria-label="Dismiss note ' +
        escapeHtml(i + 1) + ' of ' + escapeHtml(items.length) + '">Dismiss</button></li>').join("") +
      '</ul></div>';
    if (focusAt !== undefined) {
      const btns = host.querySelectorAll('button[data-note]:not([data-note="all"])');
      const b = btns[Math.min(focusAt, btns.length - 1)];
      if (b) b.focus();
    }
  }
  host.onclick = (ev) => {
    const b = ev.target.closest("button[data-note]");
    if (!b) return;
    if (b.dataset.note === "all") items.length = 0;
    else items.splice(Number(b.dataset.note), 1);
    draw(b.dataset.note === "all" ? 0 : Number(b.dataset.note));
  };
  draw();
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
