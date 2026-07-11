(() => {
  const phone = window.matchMedia("(max-width: 640px)");

  function syncDisclosures() {
    document.querySelectorAll("details[data-mobile-disclosure]").forEach((details) => {
      details.open = !phone.matches;
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", syncDisclosures, { once: true });
  } else {
    syncDisclosures();
  }
  phone.addEventListener?.("change", syncDisclosures);
})();
