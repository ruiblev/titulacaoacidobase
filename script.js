// DOM Elements
const canvas = document.getElementById('titrationChart');
const ctx = canvas.getContext('2d');
const beakerLiquid = document.getElementById('beaker-liquid');
const indicatorName = document.getElementById('indicator-name');
const currentPhDisplay = document.getElementById('current-ph');
const volAddedDisplay = document.getElementById('vol-added');

// Controls
const acidTypeSelect = document.getElementById('acid-type');
const acidVolInput = document.getElementById('acid-vol');
const acidConcInput = document.getElementById('acid-conc');
const baseConcInput = document.getElementById('base-conc');
const valAcidVol = document.getElementById('val-acid-vol');
const valAcidConc = document.getElementById('val-acid-conc');
const valBaseConc = document.getElementById('val-base-conc');

// Buttons
const btnAdd01 = document.getElementById('btn-add-01');
const btnAdd1 = document.getElementById('btn-add-1');
const btnAuto = document.getElementById('btn-auto');
const btnReset = document.getElementById('btn-reset');

// State
let state = {
    acidType: 'HCl', // 'HCl' or 'CH3COOH'
    acidVol: 50, // mL
    acidConc: 0.1, // M
    baseConc: 0.1, // M
    volBaseAdded: 0, // mL
    isAutoTitrating: false,
    intervalId: null
};

// Constants
const KW = 1.0e-14;
const Ka_CH3COOH = 1.8e-5;
const pKa_CH3COOH = -Math.log10(Ka_CH3COOH);

// Chart.js Setup
let titrationChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Curva de Titulação',
            data: [],
            borderColor: '#2196F3',
            backgroundColor: 'rgba(33, 150, 243, 0.1)',
            borderWidth: 2,
            pointRadius: 1,
            fill: false,
            tension: 0.4
        }]
    },
    options: {
        responsive: true,
        scales: {
            x: {
                type: 'linear',
                position: 'bottom',
                title: {
                    display: true,
                    text: 'Volume de Base Adicionado (mL)'
                },
                min: 0,
                max: 100 // Dynamic
            },
            y: {
                title: {
                    display: true,
                    text: 'pH'
                },
                min: 0,
                max: 14
            }
        },
        plugins: {
            tooltip: {
                callbacks: {
                    label: function (context) {
                        return `pH: ${context.parsed.y.toFixed(2)}`;
                    }
                }
            }
        }
    }
});

// Update input displays
function updateDisplays() {
    valAcidVol.textContent = acidVolInput.value;
    valAcidConc.textContent = acidConcInput.value;
    valBaseConc.textContent = baseConcInput.value;
}

[acidVolInput, acidConcInput, baseConcInput].forEach(input => {
    input.addEventListener('input', () => {
        updateDisplays();
        resetTitration();
    });
});
acidTypeSelect.addEventListener('change', resetTitration);

// Calculation Logic
function calculatePH(volBase) {
    const Va = state.acidVol / 1000; // L
    const Ca = state.acidConc;
    const Vb = volBase / 1000; // L
    const Cb = state.baseConc;
    const Vt = Va + Vb;

    if (state.acidType === 'HCl') {
        // Strong Acid + Strong Base
        const molH = Va * Ca;
        const molOH = Vb * Cb;

        if (molH > molOH) {
            const concH = (molH - molOH) / Vt;
            return -Math.log10(concH);
        } else if (molOH > molH) {
            const concOH = (molOH - molH) / Vt;
            return 14 + Math.log10(concOH);
        } else {
            return 7.00;
        }
    } else if (state.acidType === 'CH3COOH') {
        // Weak Acid + Strong Base
        // Using approximation for simulation
        const molHA = Va * Ca;
        const molOH = Vb * Cb;

        if (volBase === 0) {
            // Initial pH for Weak Acid: pH = 0.5 * (pKa - log Ca)
            return 0.5 * (pKa_CH3COOH - Math.log10(Ca));
        }

        if (molOH < molHA) {
            // Buffer Region
            // pH = pKa + log([A-]/[HA])
            const molA_minus = molOH;
            const molHA_remain = molHA - molOH;
            return pKa_CH3COOH + Math.log10(molA_minus / molHA_remain);
        } else if (Math.abs(molOH - molHA) < 1e-9) { // At equivalence (allow floating point tolerance)
            // Hydrolysis of Salt (CH3COONa)
            // pOH = 0.5 * (pKb - log C_salt)
            // pKb = 14 - pKa
            // C_salt = molHA / Vt
            const pKb = 14 - pKa_CH3COOH;
            const C_salt = molHA / Vt;
            const pOH = 0.5 * (pKb - Math.log10(C_salt));
            return 14 - pOH;
        } else {
            // Excess Base
            // Similar to strong acid, excess OH dominates
            // But we should account for the equilibrium, technically.
            // For simulation visual, Simple excess OH is usually close enough past eq point.
            /* Better approx: [OH-] = (molOH - molHA) / Vt */
            const concOH = (molOH - molHA) / Vt;
            return 14 + Math.log10(concOH);
        }

    }
    return 7;
}

// Indicator Logic (Phenolphthalein)
function getIndicatorColor(pH) {
    // Phenolphthalein: < 8.2 Colorless, > 10.0 Pink
    // We will simulate a gradient
    if (pH < 8.2) return 'rgba(255, 255, 255, 0.4)'; // Clear/White
    if (pH >= 10.0) return 'rgba(255, 105, 180, 0.8)'; // Pink

    // Gradient logic between 8.2 and 10.0
    const ratio = (pH - 8.2) / (10.0 - 8.2);
    // Mix white and pink
    return `rgba(255, 105, 180, ${0.4 + (ratio * 0.4)})`;
}

// Simulation Steps
function addTitrant(amount) {
    state.volBaseAdded += amount;
    updateSimulation();
}

function updateSimulation() {
    // 1. Calculate pH
    const pH = calculatePH(state.volBaseAdded);

    // 2. Update UI text
    currentPhDisplay.textContent = pH.toFixed(2);
    volAddedDisplay.textContent = state.volBaseAdded.toFixed(2);

    // 3. Update Chart
    titrationChart.data.labels.push(state.volBaseAdded);
    titrationChart.data.datasets[0].data.push({ x: state.volBaseAdded, y: pH });
    titrationChart.update('none'); // Efficient update

    // 4. Update Visuals
    // Liquid level in beaker increases
    // Max volume cap for visual to prevent overflow (e.g., 200mL visual cap)
    const initialHeightPerc = 30; // 30% filled at start
    const maxVolVisual = 150; // Arbitrary visual scale
    const addedHeightPerc = (state.volBaseAdded / maxVolVisual) * 50;
    let newHeight = initialHeightPerc + addedHeightPerc;
    if (newHeight > 90) newHeight = 90;

    beakerLiquid.style.height = `${newHeight}%`;
    beakerLiquid.style.backgroundColor = getIndicatorColor(pH);
}

function resetTitration() {
    clearInterval(state.intervalId);
    state.isAutoTitrating = false;
    btnAuto.textContent = "Titular Auto";

    // Read current inputs
    state.acidType = acidTypeSelect.value;
    state.acidVol = parseFloat(acidVolInput.value);
    state.acidConc = parseFloat(acidConcInput.value);
    state.baseConc = parseFloat(baseConcInput.value);
    state.volBaseAdded = 0;

    // Reset Chart
    titrationChart.data.labels = [];
    titrationChart.data.datasets[0].data = [];

    // Set X Scale Max roughly
    // Equiv point = Va * Ca / Cb. Let's make graph range 2x equiv point.
    const equivVol = (state.acidVol * state.acidConc) / state.baseConc;
    titrationChart.options.scales.x.max = Math.ceil(equivVol * 2 / 10) * 10;

    titrationChart.update();

    // Initial State Calculation
    updateSimulation();
}

function toggleAutoTitrate() {
    if (state.isAutoTitrating) {
        clearInterval(state.intervalId);
        state.isAutoTitrating = false;
        btnAuto.textContent = "Titular Auto";
    } else {
        state.isAutoTitrating = true;
        btnAuto.textContent = "Parar";
        state.intervalId = setInterval(() => {
            if (state.volBaseAdded >= titrationChart.options.scales.x.max) {
                clearInterval(state.intervalId);
                state.isAutoTitrating = false;
                btnAuto.textContent = "Titular Auto";
                return;
            }
            addTitrant(0.5); // 0.5 mL steps
        }, 100);
    }
}

// Event Listeners
btnAdd01.addEventListener('click', () => addTitrant(0.1));
btnAdd1.addEventListener('click', () => addTitrant(1.0));
btnReset.addEventListener('click', resetTitration);
btnAuto.addEventListener('click', toggleAutoTitrate);

// Init
resetTitration();
