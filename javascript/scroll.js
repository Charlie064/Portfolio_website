document.addEventListener("DOMContentLoaded", () => {
    const scrollBtn = document.querySelector(".scroll-indicator");
    const nextSection = document.querySelector(".oscilloscope-section");

    scrollBtn.addEventListener("click", () => {
        nextSection.scrollIntoView({
            behavior: "smooth"
        });
    });
});