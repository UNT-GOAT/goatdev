if (window.location.hash) {
  history.replaceState(null, "", window.location.pathname);
  window.scrollTo(0, 0);
}

const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("v");
        observer.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.06, rootMargin: "0px 0px -40px 0px" },
);

document.querySelectorAll(".sr").forEach((entry) => observer.observe(entry));
