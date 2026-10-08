/**
 * Fake News Detection in Social Media Using Machine Learning
 * File: script.js
 * Frontend interactivity, AJAX prediction requests, and sample news injection
 */

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const newsInput = document.getElementById("newsInput");
  const analyzeForm = document.getElementById("analyzeForm");
  const btnAnalyze = document.getElementById("btnAnalyze");
  const analyzeSpinner = document.getElementById("analyzeSpinner");
  const btnClear = document.getElementById("btnClear");
  const btnSampleReal = document.getElementById("btnSampleReal");
  const btnSampleFake = document.getElementById("btnSampleFake");
  const charWordCounter = document.getElementById("charWordCounter");
  const errorAlert = document.getElementById("errorAlert");

  // Theme elements
  const themeToggleBtn = document.getElementById("themeToggleBtn");
  const themeIcon = document.getElementById("themeIcon");
  const themeLabel = document.getElementById("themeLabel");

  // Theme Management
  const savedTheme = localStorage.getItem("flask_fake_news_theme") || "dark";
  applyTheme(savedTheme);

  if (themeToggleBtn) {
    themeToggleBtn.addEventListener("click", () => {
      const currentTheme = document.documentElement.getAttribute("data-theme") || "dark";
      const newTheme = currentTheme === "dark" ? "light" : "dark";
      applyTheme(newTheme);
      localStorage.setItem("flask_fake_news_theme", newTheme);
    });
  }

  function applyTheme(t) {
    if (t === "light") {
      document.documentElement.setAttribute("data-theme", "light");
      if (themeIcon) themeIcon.textContent = "🌙";
      if (themeLabel) themeLabel.textContent = "Dark Mode";
    } else {
      document.documentElement.removeAttribute("data-theme");
      if (themeIcon) themeIcon.textContent = "☀️";
      if (themeLabel) themeLabel.textContent = "Light Mode";
    }
  }

  // Output elements
  const placeholderState = document.getElementById("placeholderState");
  const resultDetails = document.getElementById("resultDetails");
  const statusDot = document.getElementById("statusDot");
  const verdictBanner = document.getElementById("verdictBanner");
  const verdictIcon = document.getElementById("verdictIcon");
  const verdictText = document.getElementById("verdictText");
  const confValue = document.getElementById("confValue");
  const meterFill = document.getElementById("meterFill");
  const meterSplit = document.getElementById("meterSplit");
  const explanationText = document.getElementById("explanationText");
  const resModel = document.getElementById("resModel");
  const resWords = document.getElementById("resWords");
  const resLength = document.getElementById("resLength");
  const resTimestamp = document.getElementById("resTimestamp");
  const tokenCloud = document.getElementById("tokenCloud");

  // Verified Samples
  const SAMPLE_REAL = `The Federal Reserve announced on Wednesday that it will maintain the benchmark interest rate between 5.25% and 5.50%. Federal Reserve Chairman Jerome Powell stated that committee members are monitoring ongoing employment figures and consumer price index trends before considering future policy rate reductions. Economic analysts noted that sustained disinflation in consumer goods has lowered immediate recession risks.`;

  const SAMPLE_FAKE = `SHOCKING: Secret government cure for all cancers discovered in lemon peels! Doctors and pharmaceutical companies are terrified! An anonymous whistleblower has revealed that boiling lemon peels with baking soda cures all stage 4 cancers in 48 hours. Big Pharma is desperately trying to delete this post from the internet! Share immediately before social media bans this page!`;

  // Update live character and word counter
  function updateCounts() {
    const text = newsInput.value.trim();
    const chars = newsInput.value.length;
    const words = text ? text.split(/\s+/).length : 0;
    charWordCounter.textContent = `${words} words · ${chars} chars`;
  }

  newsInput.addEventListener("input", updateCounts);

  // Sample Buttons
  btnSampleReal.addEventListener("click", () => {
    newsInput.value = SAMPLE_REAL;
    hideError();
    updateCounts();
    newsInput.focus();
  });

  btnSampleFake.addEventListener("click", () => {
    newsInput.value = SAMPLE_FAKE;
    hideError();
    updateCounts();
    newsInput.focus();
  });

  btnClear.addEventListener("click", () => {
    newsInput.value = "";
    hideError();
    updateCounts();
    resetResultUI();
    newsInput.focus();
  });

  function showError(msg) {
    errorAlert.textContent = msg;
    errorAlert.style.display = "block";
  }

  function hideError() {
    errorAlert.textContent = "";
    errorAlert.style.display = "none";
  }

  function resetResultUI() {
    placeholderState.style.display = "flex";
    resultDetails.style.display = "none";
    statusDot.textContent = "Awaiting input";
    statusDot.style.color = "var(--text-muted)";
  }

  // Handle Form Submission
  analyzeForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    hideError();

    const rawText = newsInput.value.trim();
    const words = rawText.split(/\s+/).filter(Boolean);

    if (words.length < 3) {
      showError("Please enter at least 3 words to perform a reliable analysis.");
      return;
    }

    // Set Loading State
    btnAnalyze.disabled = true;
    analyzeSpinner.style.display = "inline-block";
    statusDot.textContent = "Classifying text...";

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ news_text: rawText })
      });

      const data = await response.json();

      if (!response.ok || data.status === "error") {
        throw new Error(data.message || "Failed to analyze news.");
      }

      // Render Results
      renderPrediction(data);

    } catch (err) {
      showError(err.message || "Network error. Make sure the Flask server is running.");
      resetResultUI();
    } finally {
      btnAnalyze.disabled = false;
      analyzeSpinner.style.display = "none";
    }
  });

  function renderPrediction(data) {
    placeholderState.style.display = "none";
    resultDetails.style.display = "block";

    const isFake = data.prediction_code === 1 || data.prediction.includes("FAKE");
    statusDot.textContent = `Analyzed: ${data.prediction}`;
    statusDot.style.color = isFake ? "var(--fake-color)" : "var(--real-color)";

    // Update Verdict Banner
    verdictBanner.className = `verdict-banner ${isFake ? "fake-verdict" : "real-verdict"}`;
    verdictIcon.textContent = isFake ? "⚠️" : "✅";
    verdictText.textContent = data.prediction;
    confValue.textContent = `${data.confidence}%`;

    // Update Meter
    const realProb = data.probabilities ? data.probabilities.real : (isFake ? 100 - data.confidence : data.confidence);
    const fakeProb = data.probabilities ? data.probabilities.fake : (isFake ? data.confidence : 100 - data.confidence);

    meterSplit.textContent = `Real: ${realProb}% | Fake: ${fakeProb}%`;
    meterFill.className = `meter-fill ${isFake ? "fake-fill" : "real-fill"}`;
    meterFill.style.width = `${data.confidence}%`;

    // Metadata & Explanation
    explanationText.textContent = data.explanation;
    resModel.textContent = data.model_used;
    resWords.textContent = data.word_count;
    resLength.textContent = data.text_length;
    resTimestamp.textContent = data.timestamp;

    // Token Cloud
    tokenCloud.innerHTML = "";
    if (data.matched_keywords && data.matched_keywords.length > 0) {
      data.matched_keywords.forEach(token => {
        const span = document.createElement("span");
        span.className = "token-tag";
        span.textContent = token;
        tokenCloud.appendChild(span);
      });
    } else {
      const span = document.createElement("span");
      span.className = "token-tag";
      span.textContent = "general vocabulary";
      tokenCloud.appendChild(span);
    }
  }

  // Initial counter
  updateCounts();
});
