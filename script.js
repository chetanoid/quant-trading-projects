const revealTargets = document.querySelectorAll(
  ".section, .hero-panel, .stat-card, .focus-card, .project-card, .contact-panel"
);

const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.18 }
);

revealTargets.forEach((element) => {
  element.classList.add("reveal");
  observer.observe(element);
});
