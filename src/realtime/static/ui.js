/*
 * Shared UI helpers for the Voice AI Model web pages:
 * theme toggle, toast notifications, status pills and small DOM utilities.
 * Exposed as window.UI so page scripts stay framework-free.
 */
(function () {
  "use strict";

  const THEME_KEY = "voiceai-theme";

  function safeGet(key) {
    try {
      return window.localStorage.getItem(key);
    } catch (err) {
      return null;
    }
  }

  function safeSet(key, value) {
    try {
      window.localStorage.setItem(key, value);
    } catch (err) {
      /* storage unavailable (private mode); ignore */
    }
  }

  function currentTheme() {
    const explicit = document.documentElement.dataset.theme;
    if (explicit === "light" || explicit === "dark") return explicit;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  const SUN =
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>';
  const MOON =
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>';

  function renderThemeToggle(btn) {
    const theme = currentTheme();
    const next = theme === "dark" ? "light" : "dark";
    btn.innerHTML = theme === "dark" ? SUN : MOON;
    btn.setAttribute("aria-label", `Switch to ${next} theme`);
    btn.title = `Switch to ${next} theme`;
  }

  function initThemeToggle() {
    document.querySelectorAll("[data-theme-toggle]").forEach((btn) => {
      renderThemeToggle(btn);
      btn.addEventListener("click", () => {
        const next = currentTheme() === "dark" ? "light" : "dark";
        document.documentElement.dataset.theme = next;
        safeSet(THEME_KEY, next);
        renderThemeToggle(btn);
        document.dispatchEvent(new CustomEvent("themechange", { detail: next }));
      });
    });
  }

  function toastRegion() {
    let region = document.getElementById("toast-region");
    if (!region) {
      region = document.createElement("div");
      region.id = "toast-region";
      region.className = "toast-region";
      region.setAttribute("role", "status");
      region.setAttribute("aria-live", "polite");
      document.body.appendChild(region);
    }
    return region;
  }

  /** Show a transient notification. type: info | success | warning | error */
  function toast(message, type = "info", timeout = 5000) {
    const el = document.createElement("div");
    el.className = `toast ${type}`;
    if (type === "error") el.setAttribute("role", "alert");
    el.textContent = message;
    toastRegion().appendChild(el);
    window.setTimeout(() => el.remove(), timeout);
    return el;
  }

  /**
   * Update a status pill. The pill element should contain a `.dot` and a
   * `[data-value]` child. state: ok | warn | bad | busy | "" (neutral)
   */
  function setPill(pill, text, state) {
    if (!pill) return;
    const value = pill.querySelector("[data-value]");
    if (value) value.textContent = text;
    if (state) pill.dataset.state = state;
    else delete pill.dataset.state;
  }

  function escapeHtml(text) {
    const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" };
    return String(text ?? "").replace(/[&<>"']/g, (m) => map[m]);
  }

  function normalizeBackendUrl(raw) {
    let url = (raw || "").trim();
    if (!url) url = window.location.origin;
    if (!/^https?:\/\//i.test(url)) url = `${window.location.protocol}//${url}`;
    return url.replace(/\/+$/, "");
  }

  function isValidBackendUrl(raw) {
    try {
      if (/\s/.test((raw || "").trim())) return false;
      const url = new URL(normalizeBackendUrl(raw));
      return (url.protocol === "http:" || url.protocol === "https:") && /^[a-z0-9.\-[\]:]+$/i.test(url.hostname);
    } catch (err) {
      return false;
    }
  }

  /** True when the page can use getUserMedia (secure context). */
  function micAvailable() {
    return Boolean(window.isSecureContext && navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  }

  /** Human-friendly explanation of a getUserMedia failure. */
  function describeMicError(err) {
    const name = err && err.name;
    if (!micAvailable()) {
      return "Microphone access needs HTTPS or localhost. Open this page over https:// or through an SSH tunnel to localhost.";
    }
    if (name === "NotAllowedError" || name === "SecurityError") {
      return "Microphone permission was denied. Allow microphone access for this site in your browser settings, then try again.";
    }
    if (name === "NotFoundError" || name === "OverconstrainedError") {
      return "No microphone was found. Connect a microphone and try again.";
    }
    if (name === "NotReadableError") {
      return "The microphone is in use by another application. Close it and try again.";
    }
    return `Could not start the microphone${err && err.message ? `: ${err.message}` : "."}`;
  }

  /** Bind a backend URL input with validation + persistence. */
  function bindBackendInput(input, storageKey, onChange) {
    const errorEl = document.getElementById(input.getAttribute("aria-describedby")?.split(" ").find((id) => id.endsWith("-error")) || "");
    const params = new URLSearchParams(window.location.search);
    input.value = normalizeBackendUrl(params.get("backend") || safeGet(storageKey) || window.location.origin);

    input.addEventListener("change", () => {
      if (!isValidBackendUrl(input.value)) {
        input.setAttribute("aria-invalid", "true");
        if (errorEl) errorEl.textContent = "Enter a valid URL, for example http://localhost:8080";
        return;
      }
      input.removeAttribute("aria-invalid");
      if (errorEl) errorEl.textContent = "";
      input.value = normalizeBackendUrl(input.value);
      safeSet(storageKey, input.value);
      if (onChange) onChange(input.value);
    });
    return () => normalizeBackendUrl(input.value);
  }

  function markActiveNav() {
    const path = window.location.pathname.replace(/\/+$/, "") || "/";
    document.querySelectorAll(".app-nav a").forEach((a) => {
      const href = a.getAttribute("href");
      if (href === path || (href !== "/" && path.startsWith(href))) a.setAttribute("aria-current", "page");
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    initThemeToggle();
    markActiveNav();
  });

  /** Join a server error message into a sentence without doubled punctuation. */
  function sentence(text) {
    return String(text || "").trim().replace(/[.!?]+$/, "") + ".";
  }

  window.UI = {
    sentence,
    toast,
    setPill,
    escapeHtml,
    normalizeBackendUrl,
    isValidBackendUrl,
    bindBackendInput,
    micAvailable,
    describeMicError,
    storage: { get: safeGet, set: safeSet },
  };
})();
