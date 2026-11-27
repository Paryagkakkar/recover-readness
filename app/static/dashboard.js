const form = document.querySelector("#readings");
const exampleButton = document.querySelector("#load-example");
const submitButton = form.querySelector('button[type="submit"]');
const inputNote = document.querySelector("#input-note");
const formError = document.querySelector("#form-error");
const resultNote = document.querySelector("#result-note");

const example = {
  readiness: 8,
  sleep_duration_h: 7.5,
  sleep_quality: 4,
  resting_hr: 58,
  steps: 8000,
};

function showScore(kind, score) {
  const marker = document.querySelector(`#${kind}-marker`);
  document.querySelector(`#${kind}-score`).textContent = score === null ? "—" : score.toFixed(1);
  marker.hidden = score === null;
  if (score !== null) {
    marker.style.left = `${score * 10}%`;
  }
}

function resetErrors() {
  formError.hidden = true;
  form.querySelectorAll('[aria-invalid="true"]').forEach((input) => {
    input.removeAttribute("aria-invalid");
    input.classList.remove("is-invalid");
  });
}

exampleButton.addEventListener("click", () => {
  Object.entries(example).forEach(([name, value]) => {
    form.elements.namedItem(name).value = value;
  });
  resetErrors();
  inputNote.textContent = "Example loaded.";
  if (document.querySelector("#simple-score").textContent !== "—") {
    resultNote.textContent = "Showing the last estimate. Submit again for these readings.";
  }
});

form.addEventListener("input", () => {
  resetErrors();
  inputNote.textContent = "";
  if (document.querySelector("#simple-score").textContent !== "—") {
    resultNote.textContent = "Showing the last estimate. Submit again for these readings.";
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  resetErrors();
  submitButton.disabled = true;
  exampleButton.disabled = true;
  const readings = Object.fromEntries(
    Array.from(new FormData(form), ([key, value]) => [key, Number(value)])
  );
  const fields = Object.keys(readings).map((key) => form.elements.namedItem(key));
  fields.forEach((field) => { field.disabled = true; });
  submitButton.textContent = "Estimating…";

  try {
    const response = await fetch(form.action, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(readings),
    });
    const data = await response.json();
    if (!response.ok) {
      Object.keys(data.errors ?? {}).forEach((key) => {
        const field = form.elements.namedItem(key);
        field.setAttribute("aria-invalid", "true");
        field.classList.add("is-invalid");
      });
      throw new Error(data.message ?? "The estimate could not be calculated.");
    }

    showScore("simple", data.simple_estimate);
    showScore("model", data.model_estimate);
    resultNote.textContent = data.model_estimate === null
      ? "Add PMData locally to enable the model estimate."
      : "Both estimates are for the readings you submitted.";
  } catch (error) {
    formError.textContent = error.message || "The estimate could not be calculated. Try again.";
    formError.hidden = false;
  } finally {
    fields.forEach((field) => { field.disabled = false; });
    submitButton.disabled = false;
    exampleButton.disabled = false;
    submitButton.textContent = "Estimate readiness";
  }
});
