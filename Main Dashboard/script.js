document.addEventListener("DOMContentLoaded", function() {
    const cards = document.querySelectorAll(".appointment-card, .report-card, .prescription-card");

    const fadeInOnScroll = () => {
        cards.forEach(card => {
            const rect = card.getBoundingClientRect();
            if (rect.top < window.innerHeight - 50) {
                card.classList.add("card-visible");
            }
        });
    };

    window.addEventListener("scroll", fadeInOnScroll);
    fadeInOnScroll(); // Initial check
});