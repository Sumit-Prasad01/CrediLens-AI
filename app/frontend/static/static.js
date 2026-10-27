(() => {
    const form = document.getElementById("riskForm");
    const submitBtn = document.getElementById("submitBtn");
    const errorNote = document.getElementById("errorNote");
    const verdict = document.getElementById("verdict");

    const incomeInput = document.getElementById("person_income");
    const amountInput = document.getElementById("loan_amnt");
    const percentInput = document.getElementById("loan_percent_income");
    const ratioStatusBadge = document.getElementById("ratioStatusBadge");

    const gaugeFill = document.getElementById("gaugeFill");
    const gaugeThreshold = document.getElementById("gaugeThreshold");
    const probNumber = document.getElementById("probNumber");
    const stampBadge = document.getElementById("stampBadge");
    const stampText = document.getElementById("stampText");
    const stampDesc = document.getElementById("stampDesc");
    const verdictStatusPill = document.getElementById("verdictStatusPill");

    const factProb = document.getElementById("factProb");
    const factThreshold = document.getElementById("factThreshold");
    const factResult = document.getElementById("factResult");
    const factCategory = document.getElementById("factCategory");

    const apiDot = document.getElementById("apiDot");
    const apiStatusText = document.getElementById("apiStatusText");

    const GAUGE_CIRCUMFERENCE = 540.35; // 2 * PI * 86, matching SVG radius

    // ---------- Auto-calculate loan-to-income ratio ----------
    function recalcPercent() {
        const income = parseFloat(incomeInput.value);
        const amount = parseFloat(amountInput.value);
        if (income > 0 && amount >= 0) {
            const ratio = amount / income;
            percentInput.value = ratio.toFixed(2);
            if (ratioStatusBadge) {
                const pct = (ratio * 100).toFixed(0);
                ratioStatusBadge.textContent = `${pct}% of income`;
                if (ratio > 0.40) {
                    ratioStatusBadge.style.color = "var(--risk-high)";
                } else if (ratio > 0.25) {
                    ratioStatusBadge.style.color = "var(--risk-mod)";
                } else {
                    ratioStatusBadge.style.color = "var(--risk-low)";
                }
            }
        }
    }
    incomeInput.addEventListener("input", recalcPercent);
    amountInput.addEventListener("input", recalcPercent);
    recalcPercent();

    // ---------- Service status check ----------
    fetch("/openapi.json", { method: "GET" })
        .then((res) => {
            if (res.ok) {
                apiDot.className = "status-dot ok";
                apiStatusText.textContent = "Service Online";
            } else {
                throw new Error("Bad status");
            }
        })
        .catch(() => {
            apiDot.className = "status-dot down";
            apiStatusText.textContent = "Service Offline";
        });

    // ---------- Demo Preset Fillers ----------
    const presetPrime = document.getElementById("presetPrime");
    const presetSubprime = document.getElementById("presetSubprime");
    const presetReset = document.getElementById("presetReset");

    if (presetPrime) {
        presetPrime.addEventListener("click", () => {
            document.getElementById("person_age").value = 32;
            incomeInput.value = 850000;
            document.getElementById("person_home_ownership").value = "MORTGAGE";
            document.getElementById("person_emp_length").value = 7.0;
            document.getElementById("loan_intent").value = "HOMEIMPROVEMENT";
            document.getElementById("loan_grade").value = "A";
            amountInput.value = 100000;
            document.getElementById("loan_int_rate").value = 7.5;
            document.getElementById("cb_person_default_on_file").value = "N";
            document.getElementById("cb_person_cred_hist_length").value = 8;
            recalcPercent();
            clearError();
        });
    }

    if (presetSubprime) {
        presetSubprime.addEventListener("click", () => {
            document.getElementById("person_age").value = 22;
            incomeInput.value = 180000;
            document.getElementById("person_home_ownership").value = "RENT";
            document.getElementById("person_emp_length").value = 0.5;
            document.getElementById("loan_intent").value = "DEBTCONSOLIDATION";
            document.getElementById("loan_grade").value = "F";
            amountInput.value = 150000;
            document.getElementById("loan_int_rate").value = 18.5;
            document.getElementById("cb_person_default_on_file").value = "Y";
            document.getElementById("cb_person_cred_hist_length").value = 2;
            recalcPercent();
            clearError();
        });
    }

    if (presetReset) {
        presetReset.addEventListener("click", () => {
            document.getElementById("person_age").value = 30;
            incomeInput.value = 600000;
            document.getElementById("person_home_ownership").value = "RENT";
            document.getElementById("person_emp_length").value = 5.0;
            document.getElementById("loan_intent").value = "PERSONAL";
            document.getElementById("loan_grade").value = "B";
            amountInput.value = 100000;
            document.getElementById("loan_int_rate").value = 11.5;
            document.getElementById("cb_person_default_on_file").value = "N";
            document.getElementById("cb_person_cred_hist_length").value = 6;
            recalcPercent();
            clearError();
            if (verdict) verdict.hidden = true;
        });
    }

    // ---------- Visual Artifacts Tab Switcher ----------
    const tabBtns = document.querySelectorAll(".tab-btn");
    const visualImage = document.getElementById("visualImage");

    tabBtns.forEach((btn) => {
        btn.addEventListener("click", () => {
            tabBtns.forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");
            const imgFilename = btn.dataset.img;
            if (visualImage && imgFilename) {
                visualImage.src = `/visuals/${imgFilename}`;
                visualImage.alt = btn.textContent;
            }
        });
    });

    // ---------- Helpers ----------
    function setLoading(isLoading) {
        submitBtn.disabled = isLoading;
        submitBtn.classList.toggle("loading", isLoading);
        const label = submitBtn.querySelector(".btn-label");
        if (label) {
            label.textContent = isLoading ? "Underwriting Risk Analysis…" : "Assess Credit Risk";
        }
    }

    function showError(message) {
        errorNote.textContent = message;
        errorNote.hidden = false;
        errorNote.style.display = "flex";
    }

    function clearError() {
        errorNote.hidden = true;
        errorNote.textContent = "";
        errorNote.style.display = "none";
    }

    function animateNumber(el, from, to, duration) {
        const start = performance.now();
        function tick(now) {
            const t = Math.min(1, (now - start) / duration);
            const eased = 1 - Math.pow(1 - t, 3);
            const value = from + (to - from) * eased;
            el.textContent = value.toFixed(1);
            if (t < 1) requestAnimationFrame(tick);
            else el.textContent = to.toFixed(1);
        }
        requestAnimationFrame(tick);
    }

    function renderVerdict(data) {
        const probabilityPct = data.default_probability * 100;
        const thresholdPct = data.threshold * 100;
        const isHighRisk = data.default_prediction === 1;

        verdict.hidden = false;
        verdict.scrollIntoView({ behavior: "smooth", block: "nearest" });

        // Gauge fill
        const offset = GAUGE_CIRCUMFERENCE * (1 - probabilityPct / 100);
        gaugeFill.style.stroke = isHighRisk ? "var(--risk-high)" : "var(--risk-low)";
        requestAnimationFrame(() => {
            gaugeFill.style.strokeDashoffset = offset;
        });

        // Threshold indicator tick on the donut ring
        gaugeThreshold.style.transform = `rotate(${thresholdPct * 3.6}deg)`;

        // Animated readout
        animateNumber(probNumber, 0, probabilityPct, 800);

        // Stamp Banner
        if (stampBadge) {
            stampBadge.className = isHighRisk ? "verdict-banner verdict-banner--high" : "verdict-banner verdict-banner--low";
        }

        if (stampText) {
            stampText.textContent = isHighRisk ? "HIGH RISK • DECLINE / MANUAL REVIEW" : "LOW RISK • AUTO-APPROVED";
        }

        if (stampDesc) {
            stampDesc.textContent = isHighRisk
                ? `Default probability (${probabilityPct.toFixed(1)}%) exceeds policy threshold (${thresholdPct.toFixed(1)}%). Requires underwriter intervention.`
                : `Default probability (${probabilityPct.toFixed(1)}%) is within acceptable portfolio tolerance (${thresholdPct.toFixed(1)}%). Meets credit standards.`;
        }

        if (verdictStatusPill) {
            verdictStatusPill.textContent = isHighRisk ? "High Default Risk" : "Credit Approved";
            verdictStatusPill.style.background = isHighRisk ? "var(--risk-high-bg)" : "var(--risk-low-bg)";
            verdictStatusPill.style.color = isHighRisk ? "var(--risk-high)" : "var(--risk-low)";
        }

        // Ledger facts
        factProb.textContent = `${probabilityPct.toFixed(1)}%`;
        factThreshold.textContent = `${thresholdPct.toFixed(1)}%`;
        factResult.textContent = data.Result || (isHighRisk ? "High Risk" : "Low Risk");
        if (factCategory) {
            factCategory.textContent = data.risk_category || (isHighRisk ? "Subprime Tier" : "Prime Tier");
        }
    }

    // ---------- Form Submission ----------
    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        clearError();
        setLoading(true);

        const payload = {
            person_age: parseInt(document.getElementById("person_age").value, 10),
            person_income: parseFloat(incomeInput.value),
            person_home_ownership: document.getElementById("person_home_ownership").value,
            person_emp_length: parseFloat(document.getElementById("person_emp_length").value),
            loan_intent: document.getElementById("loan_intent").value,
            loan_grade: document.getElementById("loan_grade").value,
            loan_amnt: parseFloat(amountInput.value),
            loan_int_rate: parseFloat(document.getElementById("loan_int_rate").value),
            loan_percent_income: parseFloat(percentInput.value),
            cb_person_default_on_file: document.getElementById("cb_person_default_on_file").value,
            cb_person_cred_hist_length: parseInt(document.getElementById("cb_person_cred_hist_length").value, 10),
        };

        try {
            const res = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            if (!res.ok) {
                const body = await res.json().catch(() => null);
                const detail = body && body.detail ? JSON.stringify(body.detail) : `HTTP ${res.status}`;
                throw new Error(detail);
            }

            const data = await res.json();
            renderVerdict(data);
        } catch (err) {
            showError(`Assessment failed: ${err.message || "Ensure the FastAPI service is running."}`);
        } finally {
            setLoading(false);
        }
    });
})();
