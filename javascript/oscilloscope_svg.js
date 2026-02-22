const svg = document.querySelector(".oscilloscope-plot");
const svgNS = "http://www.w3.org/2000/svg";

const waveGroup = document.createElementNS(svgNS, "g");
waveGroup.setAttribute("class", "wave-lines");
svg.appendChild(waveGroup);

const gridGroup = document.createElementNS(svgNS, "g");
gridGroup.setAttribute("class", "grid-lines");
svg.appendChild(gridGroup);



// Wave settings
const width = 1000;
const height = 1000;
svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

const centerY = height / 2;
const amplitude = 0.7 * height / 2;
const numPeriods = 5
const frequency = 2 * Math.PI * numPeriods / width
const period = 2*Math.PI/frequency



function drawSineWave() {
    let dataPath = `M 0 ${centerY}`;
    for (let x=0; x<=width; x++) {
        const y = amplitude * -Math.sin(x * frequency) + centerY
        dataPath += ` L ${x} ${y}`;}

    const path = document.createElementNS(svgNS, "path");
    path.classList.add("wave-ch1");

    path.setAttribute("d", dataPath);
    path.setAttribute("stroke", "yellow");
    path.setAttribute("stroke-width", 5);
    path.setAttribute("vector-effect", "non-scaling-stroke");
    path.setAttribute("fill", "none");
    waveGroup.appendChild(path);
}


function drawTriangleWave() {
    let dataPath = `M 0 ${centerY}`; // start at center (y=0)

    const periodWidth = width / numPeriods; // width of one triangle period

    for (let x = 0; x <= width; x++) {
        const localX = x % periodWidth; // position within current period
        let y;

        if (localX < periodWidth / 4) {
            // 0 → max
            y = centerY - amplitude * (localX / (periodWidth / 4));
        } else if (localX < periodWidth / 2) {
            // max → 0
            y = centerY - amplitude * (1 - ((localX - periodWidth / 4) / (periodWidth / 4)));
        } else if (localX < (3 * periodWidth) / 4) {
            // 0 → min
            y = centerY + amplitude * ((localX - periodWidth / 2) / (periodWidth / 4));
        } else {
            // min → 0
            y = centerY + amplitude * (1 - ((localX - 3 * periodWidth / 4) / (periodWidth / 4)));
        }

        dataPath += ` L ${x} ${y}`;
    }

    const path = document.createElementNS(svgNS, "path");
    path.classList.add("wave-ch2");
    path.setAttribute("d", dataPath);
    path.setAttribute("stroke", "magenta");
    path.setAttribute("stroke-width", 3);
    path.setAttribute("vector-effect", "non-scaling-stroke");
    path.setAttribute("fill", "none");
    waveGroup.appendChild(path);
}

function drawBackgroundLines(numGridX=10, numGridY=10) {
// Vertical lines
for (let i = 0; i <= numGridX; i++) {
    const x = (i / numGridX) * width;
    const line = document.createElementNS(svgNS, "line");
    line.setAttribute("x1", x);
    line.setAttribute("y1", 0);
    line.setAttribute("x2", x);
    line.setAttribute("y2", height);
    line.setAttribute("stroke", "rgba(255,255,255,0.1)");
    if (i % 2 == 0) {
        line.setAttribute("stroke-width", 5);
        line.setAttribute("vector-effect", "non-scaling-stroke");   
    }
    else {
        line.setAttribute("stroke-width", 2);
        line.setAttribute("vector-effect", "non-scaling-stroke");
    }
    gridGroup.appendChild(line);
}

// Horizontal lines
for (let i = 0; i <= numGridY; i++) {
    const y = (i / numGridY) * height;
    const line = document.createElementNS(svgNS, "line");
    line.setAttribute("x1", 0);
    line.setAttribute("y1", y);
    line.setAttribute("x2", width);
    line.setAttribute("y2", y);
    line.setAttribute("stroke", "rgba(255,255,255,0.1)");
    if (i % 5 == 0) {
        line.setAttribute("stroke-width", 10);
        line.setAttribute("vector-effect", "non-scaling-stroke");
    }
    else {
        line.setAttribute("stroke-width", 3);
        line.setAttribute("vector-effect", "non-scaling-stroke");
    }
    gridGroup.appendChild(line);
}

svg.appendChild(gridGroup);
}


function animatePath(path, duration = 300) {
    const length = path.getTotalLength();

    // Clear any previous transition
    path.style.transition = "none";

    // Set up dash pattern
    path.style.strokeDasharray = length;
    path.style.strokeDashoffset = length;

    // Force reflow so styles apply immediately
    path.getBoundingClientRect();

    // Animate
    path.style.transition = `stroke-dashoffset ${duration}ms linear`;
    path.style.strokeDashoffset = "0";
}


function activateChannel(channel) {
    if (channel === 1) {
        wave1.style.opacity = "1";
        wave2.style.opacity = "0";

        animatePath(wave1, 1200);

        ch1Btn.classList.add("active");
        ch2Btn.classList.remove("active");
    } else {
        wave1.style.opacity = "0";
        wave2.style.opacity = "1";

        animatePath(wave2, 1200);

        ch1Btn.classList.remove("active");
        ch2Btn.classList.add("active");
    }
}


document.addEventListener("DOMContentLoaded", () => {

    // 1. Draw everything first
    drawSineWave();
    drawTriangleWave();
    drawBackgroundLines();

    // 2. Now select elements (they exist now)
    const ch1Btn = document.getElementById("ch1-btn");
    const ch2Btn = document.getElementById("ch2-btn");

    const wave1 = document.querySelector(".wave-ch1");
    const wave2 = document.querySelector(".wave-ch2");

 function activateChannel(channel) {
    if (channel === 1) {
        wave1.style.opacity = "1";
        wave2.style.opacity = "0";

        animatePath(wave1, 3000);

        ch1Btn.classList.add("active");
        ch2Btn.classList.remove("active");
    } else {
        wave1.style.opacity = "0";
        wave2.style.opacity = "1";

        animatePath(wave2, 3000);

        ch1Btn.classList.remove("active");
        ch2Btn.classList.add("active");
    }
}

    ch1Btn.addEventListener("click", () => activateChannel(1));
    ch2Btn.addEventListener("click", () => activateChannel(2));

    // Default state
    activateChannel(1);
});